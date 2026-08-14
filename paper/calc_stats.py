import sys
import os
import math

def q1(values):
    """lower quartile (25th percentile) using linear interpolation (method 7 / numpy default)"""
    n = len(values)
    if n == 0:
        return float('nan')
    sorted_vals = sorted(values)
    pos = 0.25 * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    txt_files = [f for f in os.listdir(target_dir)
                 if f.endswith('.txt') and not f.endswith('cal.txt')
                 and os.path.isfile(os.path.join(target_dir, f))]

    for fname in txt_files:
        in_path = os.path.join(target_dir, fname)
        values = []
        with open(in_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        values.append(float(line))
                    except ValueError:
                        pass

        if not values:
            print(f"{fname}: no valid data, skipping")
            continue

        min_val = min(values)
        q1_val = q1(values)

        base, _ = os.path.splitext(fname)
        out_path = os.path.join(target_dir, f"{base}cal.txt")
        with open(out_path, 'w') as fh:
            fh.write(f"Min: {min_val}\n")
            fh.write(f"Q1: {q1_val}\n")

        print(f"{fname}: Min={min_val}, Q1={q1_val} -> {base}cal.txt")

if __name__ == '__main__':
    main()
