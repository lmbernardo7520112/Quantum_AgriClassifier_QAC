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
 
