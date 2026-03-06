# Lessons Learned

### [2026-03-02 19:34:16] Experiment `exp-de736076` — `tool.nonexistent` FAILED
- **Error**: Tool 'tool.nonexistent' is not registered.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-02 19:34:16] Experiment `exp-e1e5b031` — `tool.train_qsvm` FAILED
- **Error**: Input validation failed for 'tool.train_qsvm'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-02 19:34:23] Experiment `exp-1f5b095d` — `tool.load_dataset` FAILED
- **Error**: [Errno 2] No such file or directory: '/nonexistent/path/invalid'
- **Action**: Investigate root cause, verify preconditions


## FAILED: VQE_PHASE1_BINARY
- Date: 2026-03-02T21:28:27-0300
- Error: Found array with 0 sample(s) (shape=(0, 12288)) while a minimum of 1 is required by PCA.
- Traceback: Traceback (most recent call last):
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 542, in <module>
    sys.exit(main())
             ^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 174, in main
    X_train_pca = pca.fit_transform(X_train)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/venv/lib/python3.12/site-packages/sklearn/utils/_set_output.py", line 316, in w


## FAILED: VQE_PHASE1_BINARY
- Date: 2026-03-02T21:32:42-0300
- Error: DatasetResult.__init__() got an unexpected keyword argument 'feature_dim'
- Traceback: Traceback (most recent call last):
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 503, in <module>
    sys.exit(main())
             ^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 142, in main
    binary_dataset = DatasetResult(
                     ^^^^^^^^^^^^^^
TypeError: DatasetResult.__init__() got an unexpected keyword argument 'feature_dim'



## FAILED: VQE_PHASE1_BINARY
- Date: 2026-03-02T21:34:57-0300
- Error: No module named 'qiskit_algorithms'
- Traceback: Traceback (most recent call last):
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 504, in <module>
    sys.exit(main())
             ^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/scripts/run_vqe_phase1.py", line 168, in main
    vqe_result = train_vqe_classifier(
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/quantum/vqe_classifier.py", line 73, in train_vqe_classifier
    from qiskit_algorithms imp


## FAILED: VQE_PHASE1_BINARY
- Date: 2026-03-02T21:36:45-0300
- Error: 'The primitive job to evaluate the energy failed!'
- Traceback: Traceback (most recent call last):
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/venv/lib/python3.12/site-packages/qiskit_algorithms/minimum_eigensolvers/vqe.py", line 326, in evaluate_energy
    estimator_result = job.result()[0]
                       ^^^^^^^^^^^^
  File "/mnt/c/Users/USER/Quantum_AgriClassifier_QAC/venv/lib/python3.12/site-packages/qiskit/primitives/primitive_job.py", line 70, in result
    self._result = self._future.result()
                   ^^^^^^^^^^^^^^^^^^^^^
 

### [2026-03-05 10:40:38] Experiment `exp-a20e40e3` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:40:38] Experiment `exp-0b5a5b4e` — `tool.load_dataset` FAILED
- **Error**: [Errno 2] No such file or directory: '/home/leonardomaximinobernardo/Downloads/quantum_camp_tech/DataSetEuroSat/EuroSAT_RGB'
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:40:38] Experiment `exp-0637c6b1` — `tool.run_baseline` FAILED
- **Error**: Input validation failed for 'tool.run_baseline'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:40:38] Experiment `exp-73d1b144` — `tool.train_qsvm` FAILED
- **Error**: Input validation failed for 'tool.train_qsvm'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:40:38] Experiment `exp-6545888e` — `tool.train_vqc` FAILED
- **Error**: Input validation failed for 'tool.train_vqc'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:40:38] Experiment `exp-e34b4830` — `tool.compare_models` FAILED
- **Error**: Input validation failed for 'tool.compare_models'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-5fb1b79c` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-76ac1813` — `tool.load_dataset` FAILED
- **Error**: [Errno 2] No such file or directory: '/home/leonardomaximinobernardo/Downloads/quantum_camp_tech/DataSetEuroSat/EuroSAT_RGB'
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-f129c827` — `tool.run_baseline` FAILED
- **Error**: Input validation failed for 'tool.run_baseline'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-599ccee6` — `tool.train_qsvm` FAILED
- **Error**: Input validation failed for 'tool.train_qsvm'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-a8ed1418` — `tool.train_vqc` FAILED
- **Error**: Input validation failed for 'tool.train_vqc'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 10:53:43] Experiment `exp-9f69cc87` — `tool.compare_models` FAILED
- **Error**: Input validation failed for 'tool.compare_models'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-72f95e29` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-c3793ad9` — `tool.load_dataset` FAILED
- **Error**: [Errno 2] No such file or directory: '/home/leonardomaximinobernardo/Downloads/quantum_camp_tech/DataSetEuroSat/EuroSAT_RGB'
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-c69ef4ac` — `tool.run_baseline` FAILED
- **Error**: Input validation failed for 'tool.run_baseline'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-93597bef` — `tool.train_qsvm` FAILED
- **Error**: Input validation failed for 'tool.train_qsvm'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-28ea8157` — `tool.train_vqc` FAILED
- **Error**: Input validation failed for 'tool.train_vqc'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 11:00:34] Experiment `exp-3bb06fa4` — `tool.compare_models` FAILED
- **Error**: Input validation failed for 'tool.compare_models'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 14:36:34] Experiment `exp-2f45d319` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 14:36:36] Experiment `exp-e918f02a` — `tool.run_baseline` FAILED
- **Error**: The number of classes has to be greater than one; got 1 class
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 14:37:31] Experiment `exp-9b40ba2f` — `tool.train_qsvm` FAILED
- **Error**: The number of classes has to be greater than one; got 1 class
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 14:37:31] Experiment `exp-8776bb26` — `tool.train_vqc` FAILED
- **Error**: No module named 'qiskit_algorithms'
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 14:37:31] Experiment `exp-f7019f2c` — `tool.compare_models` FAILED
- **Error**: Input validation failed for 'tool.compare_models'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 15:45:52] Experiment `exp-12b781c9` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 20:09:19] Experiment `exp-2bc2f47f` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 20:42:23] Experiment `exp-6c0588f6` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 21:15:28] Experiment `exp-15ce87af` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 21:16:46] Experiment `exp-e023ecdd` — `tool.train_vqe_classifier` FAILED
- **Error**: Input validation failed for 'tool.train_vqe_classifier'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 21:36:08] Experiment `exp-a07ba957` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 21:37:24] Experiment `exp-c3d8577c` — `tool.train_vqe_classifier` FAILED
- **Error**: Input validation failed for 'tool.train_vqe_classifier'.
- **Action**: Investigate root cause, verify preconditions

### [2026-03-05 22:25:45] Experiment `exp-b1a61922` — `tool.initialize_project` FAILED
- **Error**: Input validation failed for 'tool.initialize_project'.
- **Action**: Investigate root cause, verify preconditions
