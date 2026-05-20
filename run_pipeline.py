import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*70}\n  EJECUTANDO {script_name}\n{'='*70}")
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    python_exe = r"C:\ProgramData\miniconda3\python.exe"
    result = subprocess.run([python_exe, script_name], capture_output=False, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} falló con código {result.returncode}")
        sys.exit(result.returncode)

def main():
    scripts = [
        "07_cnn_clasificacion.py",
        "08_fusion_features.py",
        "10_evaluacion_comparativa.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n\n" + "="*70)
    print("  ¡PIPELINE COMPLETADO EXITOSAMENTE!")
    print("="*70)

if __name__ == "__main__":
    main()
