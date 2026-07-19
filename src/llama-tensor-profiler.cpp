    return result;
}

std::vector<std::string> llama_tensor_profiler::generate_overrides(const placement_solution & solution) const {
    std::vector<std::string> overrides;
    for (const auto & name : solution.gpu_tensors) {
        overrides.push_back(name + "=CUDA0");
    }
    for (const auto & name : solution.cpu_tensors) {
        overrides.push_back(name + "=CPU");
    }
    LLAMA_LOG_INFO("%s: generated %zu tensor overrides (%zu GPU, %zu CPU)\n",
                   __func__, overrides.size(), solution.gpu_tensors.size(),
                   solution.cpu_tensors.size());
    return overrides;
}

void llama_tensor_profiler::clear() {
    entries.clear();
    active.clear();
}
