# Results Task 7.2 Time measurement

### Run 1:

* 1 job
* Nodes used: 4
* Threads used: 12
* Epochs: 25
* Time in seconds: 369

Final Terminal Output:
```
  training loss: 521.751
  training loss reduce: 522.774
  test loss: 89.892
  test accuracy: 0.799
  test loss reduce: 91.258
  test accuracy reduce: 0.796

... Run took 369 seconds.
```

### Run 2: 

* 1 job
* Nodes used: 1
* Threads used: 12
* Epochs: 25
* Time in seconds: 892

Final Terminal Output:
```
  training loss: 529.774
  training loss reduce: 529.774
  test loss: 91.518
  test accuracy: 0.795
  test loss reduce: 91.518
  test accuracy reduce: 0.795

... Run took 892 seconds.
```

### Run 3: 

* On slurm with 4 jobs
* Nodes used: 4
* Threads used: 12
* Epochs: 25
* Time in seconds: 338

Final Terminal Output:
```
  training loss: 525.509
  training loss reduce: 523.967
  test loss: 90.335
  test accuracy: 0.798
  test loss reduce: 91.439
  test accuracy reduce: 0.797

... Run took 338 seconds.
```