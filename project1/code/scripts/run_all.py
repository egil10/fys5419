import os
import subprocess
import glob
import sys
import argparse

def run_all_scripts():
    parser = argparse.ArgumentParser(description="Run all project parts.")
    parser.add_argument("--no-show", action="store_true", help="Run without showing plot windows.")
    args = parser.parse_args()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all part-*.py files
    scripts = sorted(glob.glob(os.path.join(script_dir, "part-*.py")))
    
    # Results directory path
    results_dir = os.path.normpath(os.path.join(script_dir, "..", "results"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Found {len(scripts)} scripts. Saving output to {results_dir}")
    if args.no_show:
        print("Note: Plot windows are disabled (--no-show).")
    
    # Setup environment
    env = os.environ.copy()
    if args.no_show:
        env["MPLBACKEND"] = "Agg" # Disables GUI windows
    
    for script in scripts:
        script_name = os.path.basename(script)
        out_filename = script_name.upper().replace(".PY", "_RESULTS.TXT")
        out_path = os.path.join(results_dir, out_filename)
        
        print(f"\n>>> Running {script_name}...")
        
        try:
            # Run the script and capture output
            result = subprocess.run(["python", script], env=env, capture_output=True, text=True, check=True)
            
            # Print to console for visibility
            print(result.stdout)
            
            # Save to results folder
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
                
            print(f">>> {script_name} finished. Output saved to {out_filename}")
            
        except subprocess.CalledProcessError as e:
            print(f">>> Error running {script_name}:\n{e.stdout}\n{e.stderr}")

if __name__ == "__main__":
    run_all_scripts()
