# floff
A test repository for the WIP algorithm for fixed-precision floating-point formatting.

An explanation of the algorithm is given [here](https://jk-jeon.github.io/posts/2022/12/fixed-precision-formatting/).

For quick experiment: [https://godbolt.org/z/5948en1zo](https://godbolt.org/z/5948en1zo)


![benchmark](subproject/benchmark/results/to_chars_fixed_precision_benchmark_binary64.png)
- **Red**: Proposed algorithm with the full (9904 bytes) cache table and the long (3688 bytes) extended cache table.
- **Green**: Proposed algorithm with the compressed (584 bytes) cache table and the super-compact (580 bytes) extended cache table.
- **Blue**: Ryu-printf (reference implementation).
- **Purple**: fmtlib.

Compiled binary size of the benchmark program:
```
                                base : 1,022,464 bytes
                floff full-long only : 1,050,112 bytes (base +  27,648 bytes)
 floff compressed-super-compact only : 1,040,384 bytes (base +  17,920 bytes)
                     Ryu-printf only : 1,133,568 bytes (base + 111,104 bytes)
                above three together : 1,180,160 bytes (base + 157,696 bytes)
```
