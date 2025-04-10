### 🧩 Initial State — Global Memory (`d`)

Each thread `i` will read `d[i]`. This is the input array in **global memory**:

```
Thread i:     0   1   2   3   ...         60  61  62  63
Global d[i]:  0   1   2   3   ...         60  61  62  63
```

---

### 🧠 Step 1 — Each Thread Copies to Shared Memory `s[i]`

```
Thread i:     0   1   2   3   ...         60  61  62  63
Copy:         d[0] -> s[0]
              d[1] -> s[1]
              d[2] -> s[2]
              ...
              d[63] -> s[63]
Shared s[i]:  0   1   2   3   ...         60  61  62  63
```

Each thread reads **its own index** from global memory and writes into the same index in shared memory.

---

### ⏸️ Step 2 — Synchronize (`__syncthreads()`)

All threads wait here to ensure shared memory is **fully populated** before continuing.

```
Barrier reached ───> All threads synchronized here
```

---

### 🔄 Step 3 — Reverse Read From Shared Memory (`s[n - i - 1]`) → Write to `d[i]`

Each thread now reads from **reverse index** of shared memory and writes to **its own index** in global memory:

```
Thread i:     0     1     2     3    ...   60    61    62    63
Read from:   s[63] s[62] s[61] s[60] ...  s[3]  s[2]  s[1]  s[0]
Write to:    d[0]  d[1]  d[2]  d[3]  ...  d[60] d[61] d[62] d[63]

New d[i]:    63    62    61    60   ...  3     2     1     0
```

---

### ✅ Final Output — Global Memory is Reversed

```
Global d[i]:  63  62  61  60  ...  3  2  1  0
```

Every thread reversed its value by using shared memory to access the inverse index.
