import os
import subprocess
import glob

def run_all_scripts():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all part-*.py files
    scripts = sorted(glob.glob(os.path.join(script_dir, "part-*.py")))
    
    print(f"Found {len(scripts)} scripts. Running them sequentially...")
    
    for script in scripts:
        script_name = os.path.basename(script)
        print(f"\n>>> Running {script_name}...")
        
        # We run them using subprocess. 
        # Note: If plt.show() is called, this will wait until the user closes the window.
        try:
            result = subprocess.run(["python", script], check=True, text=True)
            print(f">>> {script_name} finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f">>> Error running {script_name}: {e}")

if __name__ == "__main__":
    run_all_scripts()
