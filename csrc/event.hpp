#include <ATen/cuda/CUDAContext.h>
#include <memory>

#include "kernels/exception.cuh"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;
    at::cuda::CUDAStream recorded_stream;

    EventHandle() : recorded_stream(at::cuda::getCurrentCUDAStream()) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(recorded_stream);
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) : recorded_stream(stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
    }
};

torch::Event create_event(const at::cuda::CUDAStream &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    if (s != event.recorded_stream)
        s.unwrap().wait(*event.event);
}

} // namespace deep_ep
