# collective-api

Supported operators 

| **Operator** | **Scalar/ Buffer** | **Table** |
|:------------:|:------------------:|:---------:|
|   Barrier    |         -          |     -     |
|  Broadcast   |         Y          |     Y     |
|  Allgather   |         Y          |     Y     |
|    Gather    |         Y          |     ?     |
|  Allreduce   |         Y          |     -     |
|   Shuffle    |         -          |     Y     |
|   Scatter    |         ?          |     Y     |