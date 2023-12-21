# collective-api

Supported operators 

| **Operator** | **Scalar/ Buffer** | **Table** |
|:------------:|:------------------:|:---------:|
|    Barrier   |          -         |     -     |
|   Allgather  |          Y         |     Y     |
|    Gather    |          Y         |     ?     |
|   Allreduce  |          Y         |     -     |
|    Shuffle   |          -         |     Y     |
|    Scatter   |          ?         |     Y     |