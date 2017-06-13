# GPU Sum Reduction

An attempt at an optimized GPU sum reduction, based on this [talk by NVIDIA's Mark Harris](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf).

## Final Results
At the last reasonable point of optimizing the implementation, the sum reduce kernel achieves about 96% of the theoretical memory bandwidth of my laptop's GPU (GeForce GTX 850M), at 13.867 GB/s versus the theoretical 14.4 GB/s. Recall that reduction is constrained mainly by memory bandwidth, since the algorithm is not compute-intensive at all. Thus, as we have acheived an excellent percentage of the theoretical memory bandwidth, we can reasonably stop at this point.

## Main Strategies
- Process as much data as possible (without affecting algorithm correctness) in shared memory.
- Use sequential addressing to get rid of bank conflicts, both on global-to-shared memory data load, and processing the shared memory data.

## Other notes
- This implementation performs better than Reduction #3 (sequential addressing) and even Reduction #4 (first add during load) in Mark Harris' slides, without going into loop unrolling (beyond Reduction #4). 

   To be more specific, the shared efficiency (as measured in nvvp / NVIDIA Visual Profiler) is higher in my implementation (96.1%) than Reduction #4 (84.6%).

   This most likely owes to the fact that my implementation processes more data in shared memory (2048 items) than Reduction #4 (1024 items).
   
- For some unknown reason (for now), incorporating loop unrolling in the implementation makes the Debug compiled code output the correct values, while the Release compiled code outputs incorrect values.
