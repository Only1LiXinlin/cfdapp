import itertools
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import networkx as nx
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='cfdmine.log',
                    filemode='w')


# Helper functions
def read_csv_in_chunks(file_path, chunk_size=10000):
    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        for chunk in chunks:
            chunk.fillna('MISSING', inplace=True)
            yield chunk.to_dict(orient='records')
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def generate_combinations(attributes, size):
    return list(itertools.combinations(attributes, size))


def calculate_support(data, lhs, rhs, condition):
    support_count = 0
    total_count = 0
    for row in data:
        if all(row.get(attr) == val for attr, val in condition.items()):
            total_count += 1
            if all(row.get(lhs_attr) == row.get(rhs) for lhs_attr in lhs):
                support_count += 1
    return support_count, total_count


def parallel_calculate_support(data, lhs, rhs, conditions):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_support, data, lhs, rhs, condition): condition for condition in conditions}
        for future in as_completed(futures):
            results.append(future.result())
    return results


# Optimized CTANE Algorithm
def ctane(data, column_names, min_support=0.5):
    attributes = column_names
    L = {1: [(attr,) for attr in attributes]}
    valid_cfds = []
    k = 1

    while L[k]:
        C_k_plus_1 = []
        for lhs_tuple in L[k]:
            lhs_attrs = [x for x in lhs_tuple]
            for rhs in attributes:
                if rhs not in lhs_attrs:
                    try:
                        conditions = [dict(zip(lhs_attrs, values))
                                      for values in
                                      itertools.product(*(set(row[attr] for row in data) for attr in lhs_attrs))]
                        results = parallel_calculate_support(data, lhs_attrs, rhs, conditions)
                        for idx, (support, total) in enumerate(results):
                            if total > 0 and support / total >= min_support:
                                C_k_plus_1.append((lhs_tuple, rhs, conditions[idx]))
                                valid_cfds.append((lhs_tuple, rhs, conditions[idx]))
                    except KeyError as e:
                        print(f"KeyError: {e} with lhs_attrs: {lhs_attrs} and rhs: {rhs}")
        L[k + 1] = C_k_plus_1
        k += 1

    return valid_cfds


# Optimized CFDMine Algorithm
def cfdmine(data, column_names, min_support=0.5):
    logging.info('Starting CFDMine algorithm')

    def get_unique_values(attribute):
        return set(row[attribute] for row in data)

    def generate_conditions(lhs):
        return [dict(zip(lhs, values)) for values in itertools.product(*(get_unique_values(attr) for attr in lhs))]

    def prune_candidates(candidates):
        pruned = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_candidate = {executor.submit(generate_and_prune, lhs, rhs): (lhs, rhs) for lhs, rhs in candidates}
            for future in as_completed(future_to_candidate):
                lhs, rhs = future_to_candidate[future]
                try:
                    pruned.extend(future.result())
                except Exception as exc:
                    logging.error(f'Error processing LHS: {lhs}, RHS: {rhs} - {exc}')
        return pruned

    def generate_and_prune(lhs, rhs):
        pruned = []
        conditions = generate_conditions(lhs)
        start_time = time.time()
        results = parallel_calculate_support(data, lhs, rhs, conditions)
        end_time = time.time()
        logging.info(f'Calculated support for LHS: {lhs}, RHS: {rhs} in {end_time - start_time:.2f} seconds')
        for idx, (support, total) in enumerate(results):
            if total > 0 and support / total >= min_support:
                pruned.append((lhs, rhs, conditions[idx]))
        return pruned

    attributes = column_names
    candidate_cfds = [(lhs, rhs) for size in range(1, len(attributes)) for lhs in
                      generate_combinations(attributes, size) for rhs in attributes if rhs not in lhs]

    start_time = time.time()
    valid_cfds = prune_candidates(candidate_cfds)
    end_time = time.time()

    logging.info(f'CFDMine algorithm completed in {end_time - start_time:.2f} seconds')
    return valid_cfds


# GUI
class CFDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conditional Functional Dependency Miner")

        self.file_path = tk.StringVar()
        self.min_support = tk.DoubleVar(value=0.5)
        self.selected_algorithm = tk.StringVar(value="CTANE")

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.file_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Browse", command=self.load_csv).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(frame, text="Min Support:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.min_support).grid(row=1, column=1, sticky=(tk.W, tk.E))

        ttk.Label(frame, text="Algorithm:").grid(row=2, column=0, sticky=tk.W)
        ttk.Combobox(frame, textvariable=self.selected_algorithm, values=["CTANE", "CFDMine", "Splitting-CFD"]).grid(
            row=2, column=1, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Run Algorithm", command=self.run_algorithm).grid(row=3, column=1, sticky=tk.E)
        ttk.Button(frame, text="Save Results", command=self.save_results).grid(row=3, column=2, sticky=tk.E)
        ttk.Button(frame, text="Visualize", command=self.visualize_results).grid(row=3, column=3, sticky=tk.E)

        self.result_text = tk.Text(frame, width=80, height=20)
        self.result_text.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E))

        self.status_label = ttk.Label(frame, text="Status: Idle")
        self.status_label.grid(row=6, column=0, columnspan=4, sticky=tk.W)

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(4, weight=1)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)
            try:
                self.data, self.column_names = self.load_csv_data(file_path)
            except Exception as e:
                messagebox.showerror("File Error", str(e))

    def load_csv_data(self, file_path):
        data = []
        column_names = None
        for chunk in read_csv_in_chunks(file_path):
            data.extend(chunk)
            if not column_names:
                column_names = chunk[0].keys()
        return data, column_names

    def run_algorithm(self):
        self.update_status("Running...")
        self.progress.start()
        min_support = self.min_support.get()
        algorithm = self.selected_algorithm.get()

        start_time = time.time()

        if algorithm == "CTANE":
            self.results = ctane(self.data, self.column_names, min_support)
        elif algorithm == "CFDMine":
            self.results = cfdmine(self.data, self.column_names, min_support)
        else:
            self.results = splitting_cfd(self.data, self.column_names, min_support)

        end_time = time.time()
        self.progress.stop()
        self.update_status(f"Completed in {end_time - start_time:.2f} seconds")

        self.display_results(self.results)

    def update_status(self, message):
        self.status_label.config(text=message)

    def display_results(self, results):
        self.result_text.delete("1.0", tk.END)
        for cfd in results:
            lhs_tuple, rhs, condition = cfd
            lhs_attrs = [x for x in lhs_tuple]
            condition_str = " AND ".join([f"{k}={v}" for k, v in condition.items()])
            self.result_text.insert(tk.END, f"IF {condition_str} THEN {', '.join(lhs_attrs)} -> {rhs}\n")

    def save_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                for cfd in self.results:
                    lhs_tuple, rhs, condition = cfd
                    lhs_attrs = [x for x in lhs_tuple]
                    condition_str = " AND ".join([f"{k}={v}" for k, v in condition.items()])
                    file.write(f"IF {condition_str} THEN {', '.join(lhs_attrs)} -> {rhs}\n")
            messagebox.showinfo("Save Results", "Results saved successfully.")

    def visualize_results(self):
        G = nx.DiGraph()
        for cfd in self.results:
            lhs_tuple, rhs, condition = cfd
            lhs_attrs = [x for x in lhs_tuple]
            for lhs in lhs_attrs:
                G.add_edge(lhs, rhs, label=str(condition))

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold',
                arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Conditional Functional Dependencies")
        plt.show()


def main():
    root = tk.Tk()
    app = CFDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
