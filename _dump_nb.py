import json, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def dump(path):
    nb = json.load(open(path, encoding='utf-8'))
    print(f"\n========== {os.path.basename(path)} ==========")
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c.get('source', []))
        src_short = '\n'.join(src.splitlines()[:8])
        outs = c.get('outputs', [])
        if not outs and len(src.strip()) < 20:
            continue
        print(f"\n--- cell {i} ---\n{src_short}")
        for o in outs:
            ot = o.get('output_type')
            if ot == 'stream':
                t = o.get('text', '')
                if isinstance(t, list): t = ''.join(t)
                print(">>", t.rstrip())
            elif ot in ('execute_result', 'display_data'):
                d = o.get('data', {})
                tp = d.get('text/plain', '')
                if isinstance(tp, list): tp = ''.join(tp)
                if tp:
                    print("=>", tp.rstrip())
                # ignore image/html
            elif ot == 'error':
                print("!!ERR:", o.get('ename'), o.get('evalue'))

for p in sys.argv[1:]:
    dump(p)
