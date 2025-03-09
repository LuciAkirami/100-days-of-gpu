## Optimizing Data Transfers

Let’s start with a few general guidelines for host-device data transfers.

-   Minimize the amount of data transferred between host and device when possible, even if that means running kernels on the GPU that get little or no speed-up compared to running them on the host CPU.
-   Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.
-   Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.
-   Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers.
