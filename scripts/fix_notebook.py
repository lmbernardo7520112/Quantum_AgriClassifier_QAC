import json
import sys

def fix_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    # Join lines
                    joined = ''.join(source)
                    # Replace print(f"\n string to avoid SyntaxError in python
                    joined = joined.replace('print(f"\\n', 'print(f"\\\\n')
                    # Also if there are literal newlines inside the f-string, fix them:
                    # In python string `print(f"\n\U0001f4ca Modelo Quântico (VQC)")`
                    # it might have been split into two lines in the JSON:
                    # `"print(f\"\n",` and `"📊 Modelo Quântico (VQC)\")\n"`
                    # We can just fix them manually for the specific prints
                    if 'print(f"\\n📊' in joined:
                        joined = joined.replace('print(f"\\n📊', 'print(f"\\\\n📊')
                        
                    # Also replace multiline literal newlines inside print(f")
                    import re
                    # Find all "print(f" followed by actual newline
                    joined = re.sub(r'print\(f"\n(.*?)', r'print(f"\\n\1', joined, flags=re.MULTILINE)
                    
                    # Also fix the one that split across lines:
                    joined = joined.replace('print(f"\n📊', 'print(f"\\\\n📊')
                    joined = joined.replace('print(f"\n📈', 'print(f"\\\\n📈')
                    joined = joined.replace('print(f"\n🔬', 'print(f"\\\\n🔬')

                    # reconstruct source
                    if '\n' in joined:
                        lines = joined.split('\n')
                        # re-add \n to the ends of all but last
                        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
                        if len(cell['source']) > 0 and cell['source'][-1] == '':
                             cell['source'].pop()
                    else:
                        cell['source'] = [joined]
                        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook fixed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    fix_notebook('notebooks/QAC_Bloco3_Experiment.ipynb')
