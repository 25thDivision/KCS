from qiskit_ibm_runtime import QiskitRuntimeService

instances = [
    # "Yonsei_internal",
    # "Y_BS_candid",
    "Yonsei_2026_1st_semester",
]

for inst in instances:
    try:
        service = QiskitRuntimeService(instance=inst)
        backends = service.backends()
        qpus = [b for b in backends if not b.simulator]
        print(f"\n=== {inst} ({len(qpus)} QPUs) ===")
        for b in qpus:
            status = b.status()
            print(f"  {b.name}: {b.num_qubits}q, "
                  f"{'online' if status.operational else 'offline'}, "
                  f"pending_jobs={status.pending_jobs}")
    except Exception as e:
        print(f"\n=== {inst}: ERROR - {e} ===")