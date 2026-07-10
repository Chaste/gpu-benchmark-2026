"""Generate results.csv from raw benchmark timings.

Reads an input file written by WriteResultsToFile() in ExampleApp_gpu-benchmark-2026.cu,
averaged across repetitions) and computes the extra columns needed by graphs.py

    Population Size               = box width ** 2 (2D) or ** 3 (3D)
    Proportion Mechanics           = Mechanics Time / Total Time
    Maximum Theoretical Speedup    = 1 / (1 - cpu's Proportion Mechanics) (Amdahl's law)
    Actual Speedup                 = cpu Total Time / gpu Total Time
    % Maximum Speedup Achieved     = Actual Speedup / Maximum Theoretical Speedup * 100
    Mechanics Speedup              = cpu Mechanics Time / gpu Mechanics Time

The four speedup-related columns only apply to gpu rows

Run as a script, optionally specifying the input/output files:

    python3 generate_results.py [-i raw-timings.csv] [-o results.csv]
"""

import argparse
import csv

DEFAULT_INPUT_FILE = 'raw-timings.csv'
DEFAULT_OUTPUT_FILE = 'results.csv'

FIELDNAMES = [
    'Simulator', 'Dimensions', 'Box width', 'Mechanics Time', 'Total Time',
    'Population Size', 'Proportion Mechanics', 'Maximum Theoretical Speedup',
    'Actual Speedup', '% Maximum Speedup Achieved', 'Mechanics Speedup',
]


def load_raw_timings(path):
    with open(path, newline='') as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                'Simulator': row['Simulator'],
                'Dimensions': row['Dimensions'],
                'Box width': float(row['Box width']),
                'Mechanics Time': float(row['Mechanics Time']),
                'Total Time': float(row['Total Time']),
            })
        return rows


def population_size(dimensions, box_width):
    return box_width ** 2 if dimensions == '2D' else box_width ** 3


def derive_results(raw_rows):
    cpu_rows = {(r['Dimensions'], r['Box width']): r for r in raw_rows if r['Simulator'] == 'cpu'}

    results = []
    for row in raw_rows:
        out = dict(row)
        out['Population Size'] = population_size(row['Dimensions'], row['Box width'])
        out['Proportion Mechanics'] = row['Mechanics Time'] / row['Total Time']

        if row['Simulator'] == 'gpu':
            cpu_row = cpu_rows[(row['Dimensions'], row['Box width'])]
            max_theoretical_speedup = 1 / (1 - cpu_row['Mechanics Time'] / cpu_row['Total Time'])
            actual_speedup = cpu_row['Total Time'] / row['Total Time']

            out['Maximum Theoretical Speedup'] = max_theoretical_speedup
            out['Actual Speedup'] = actual_speedup
            out['% Maximum Speedup Achieved'] = actual_speedup / max_theoretical_speedup * 100
            out['Mechanics Speedup'] = cpu_row['Mechanics Time'] / row['Mechanics Time']
        else:
            out['Maximum Theoretical Speedup'] = ''
            out['Actual Speedup'] = ''
            out['% Maximum Speedup Achieved'] = ''
            out['Mechanics Speedup'] = ''

        results.append(out)
    return results


def format_number(value):
    if value == '':
        return ''
    if value == int(value):
        return str(int(value))
    return f'{value:.10g}'


def write_results(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        for row in rows:
            writer.writerow([format_number(row[name]) if name not in ('Simulator', 'Dimensions')
                              else row[name] for name in FIELDNAMES])


def parse_args():
    parser = argparse.ArgumentParser(description='Generate results.csv from raw benchmark timings.')
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE,
                         help=f'raw timings CSV to read (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE,
                         help=f'results CSV to write (default: {DEFAULT_OUTPUT_FILE})')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    raw_rows = load_raw_timings(args.input)
    results = derive_results(raw_rows)
    write_results(results, args.output)
    print(f'Wrote {len(results)} rows to {args.output}')
