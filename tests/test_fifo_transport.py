"""Distributed exact transport regression; launch with torchrun (SM20)."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import deep_ep

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=12)
    a=p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank,world=dist.get_rank(),dist.get_world_size()
    a.out.mkdir(parents=True,exist_ok=True)
    # Allocate once at reference-path capacity; then test both paths on the same buffer.
    os.environ['DEEPEP_FIFO_OPTIMIZED']='0'
    b=deep_ep.Buffer(dist.group.WORLD,1984*1024**2,0,allow_mnnvl=True,use_fabric=True,explicitly_destroy=True)
    assert b.runtime.get_num_rdma_ranks()==1
    rows=[]
    try:
        for fifo in (0,1):
            os.environ['DEEPEP_FIFO_OPTIMIZED']=str(fifo)
            for hidden,tokens,route in [(3072,513,'random'),(4096,513,'random'),(7168,513,'random'),
                                         (8192,513,'random'),(3072,17,'local'),(4096,17,'remote'),
                                         (3072,17,'hotspot'),(3072,0,'empty'),(3072,4096,'random')]:
                experts=world*32
                gen=torch.Generator(device='cuda').manual_seed(20261121+rank)
                ids=torch.rand(tokens,experts,device='cuda',generator=gen).topk(8,dim=-1).indices
                if route=='local':ids=ids%32+rank*32
                if route=='remote':ids=ids%32+((rank+1)%world)*32
                if route=='hotspot':ids=ids%32
                weights=torch.ones(tokens,8,device='cuda',dtype=torch.float32)
                x=torch.full((tokens,hidden),rank+1,device='cuda',dtype=torch.bfloat16)
                layout=b.get_dispatch_layout(ids,experts,reuse_workspace=bool(fifo))
                expected=x*layout[3].sum(1,keepdim=True)
                config=deep_ep.Config(20,8,256)
                timings=[]
                for sample in range(a.samples+2):
                    # Cover asynchronous readiness, cached handles, bias and padded receives.
                    async_finish=sample%2==1
                    padded=sample%3==1
                    worst=world*tokens if padded else 0
                    ds,de,cs,ce=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    ds.record()
                    received,rids,rweights,counts,handle,event=b.dispatch(
                        x,num_tokens_per_rank=layout[0],num_tokens_per_expert=layout[2],
                        is_token_in_rank=layout[3],topk_idx=ids,topk_weights=weights,
                        config=config,num_worst_tokens=worst,async_finish=async_finish)
                    if async_finish:event.current_stream_wait()
                    de.record()
                    bias=torch.ones_like(x) if sample%3==2 else None
                    cs.record()
                    output,_,event=b.combine(received,handle,bias=bias,config=config,async_finish=async_finish)
                    if async_finish:event.current_stream_wait()
                    ce.record()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output,expected+(1 if bias is not None else 0),atol=0,rtol=0)
                    # Check payload source ownership before any expert computation.
                    prefix=handle[0].reshape(world,world)
                    start=0
                    for source in range(world):
                        stop=int(prefix[source,rank])
                        assert bool((received[start:stop]==source+1).all()),(rank,route,source)
                        start=stop
                    if sample==0:
                        cached,_,_,_,_,event=b.dispatch(x,handle=handle,config=config)
                        cached_output,_,event=b.combine(cached,handle,config=config)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(cached_output,expected,atol=0,rtol=0)
                    if sample>=2:timings.append([ds.elapsed_time(de),cs.elapsed_time(ce)])
                rows.append(dict(rank=rank,fifo=fifo,hidden=hidden,tokens=tokens,route=route,
                                 samples_ms=timings,correct=True,sms=20))
                if rank==0:print('PASS',fifo,hidden,tokens,route,flush=True)
        (a.out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        b.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
