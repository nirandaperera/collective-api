# Buffer Collectives 

Collectives on buffers 

Questions?
1. Use templates or OOP?
- OOP
2. Type of buffer? Do we need a `Buffer` class? Reuse `arrow::Buffer` class? 
How to support both CPU/ GPU buffers?
- may be not GPUs. 
- rmm device spans or host spans 
- c++20 std::spans - build eng 
3. Ownership of the send buffers?
- passed to 
4. Allocation of receive buffers?
- 
5. Futures vs CBs?
- Callbacks 

