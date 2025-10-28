import argparse
import json
import os
import random
import traceback
from functools import partial
import multiprocessing

import numpy as np
import datasets
from tqdm import tqdm

import graph as gh
import problem as pr


# A set of basic geometric objects to initiate the generation process.
PRIMITIVE_SET = [
    "segment", "angle",
    "triangle", "r_triangle", "iso_triangle", "ieq_triangle", "risos", "triangle12",
    "quadrangle", "rectangle", "isquare", "trapezoid", "r_trapezoid", "eq_trapezoid", "eq_quadrangle", "eqdia_quadrangle",
    "pentagon", "eq_pentagon",
]

# A set of geometric relations used to incrementally add complexity.
# The keys (1 or 2) indicate the number of new points introduced by the relation.
RELATION_SET = {
    1: ['angle_bisector', 'angle_mirror', 'circle', 'circumcenter', 'eq_triangle', 'eqangle2', 'eqdistance', 'foot', 'incenter', 'excenter', 'intersection_cc', 'intersection_lc', 'intersection_ll', 'intersection_lp', 'intersection_lt', 'intersection_pp', 'intersection_tt', 'lc_tangent', 'midpoint', 'mirror', 'nsquare', 'on_aline', 'on_bline', 'on_circle', 'on_line', 'on_pline', 'on_tline', 'orthocenter', 'parallelogram', 'psquare', 'reflect', 's_angle', 'shift', 'on_dia', 'on_opline', 'eqangle3', 'on_circum'],
    2: ['square', 'trisect', 'trisegment', 'tangent']
}

WORKER_DEFINITIONS = None

def init_worker(definitions_dict):
    """Initializer for each worker process in the pool."""
    global WORKER_DEFINITIONS
    WORKER_DEFINITIONS = definitions_dict

def generate_points(current_idx, n):
    """Generates a list of n new variable names, e.g., ['x0', 'x1']."""
    return [f"x{current_idx + i}" for i in range(n)]

def generate_image_and_caption(code: str, seed: int, problem_idx: int, step_num: int, image_cache_num: int, image_cache_folder: str) -> tuple:
    """Generates an image and caption for a given geometric code."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        p = pr.Problem.from_txt(code)
        g, deps, captions = gh.Graph.build_problem(p, WORKER_DEFINITIONS, verbose=False)
        caption = '\n'.join(captions)

        highlights = []
        for i, dep in enumerate(deps):
            if i > 0 and dep.name == 'aconst' and deps[i - 1].name == 'aconst':
                continue
            highlights.append((dep.name, dep.args))

        save_to_path = None
        if problem_idx < image_cache_num:
            image_path = os.path.join(image_cache_folder, str(problem_idx), f"step_{step_num}.png")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            save_to_path = image_path

        image_bytes = gh.nm.draw(
            g.type2nodes[gh.Point], g.type2nodes[gh.Line], g.type2nodes[gh.Circle],
            g.type2nodes[gh.Segment],
            highlights=highlights,
            return_bytes=True,
            save_to=save_to_path,
            seed=seed
        )
        return image_bytes, caption
    except Exception:
        traceback.print_exc()
        return None, None

def save_dataset_in_batches(dataset, split_type: str, dataset_name: str, save_path: str, batch_size: int):
    """Saves a Hugging Face dataset to Parquet files in batches."""
    save_dir = os.path.join(save_path, f'{split_type}_parquet')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving {dataset_name} {split_type} data to {save_dir}...")
    
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for i in tqdm(range(total_batches), desc=f"Saving {split_type} batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        subset = dataset.select(range(start_idx, end_idx))
        file_name = f"{dataset_name}_batch_{i:03d}.parquet"
        file_path = os.path.join(save_dir, file_name)
        subset.to_parquet(file_path)
        
    print(f"\n{dataset_name.capitalize()} {split_type} data saved successfully.")
    print(f"Total records: {len(dataset)}, saved in {total_batches} batch(es).")


def _generate_initial_clause(primitive_name: str):
    """Generates the first clause based on a primitive object."""
    primitive_def = WORKER_DEFINITIONS[primitive_name]
    
    if primitive_def.construction.name == 'angle':
        points = generate_points(0, 3)
        clauses = [f"{' '.join(points)} = angle {' '.join(points)} {random.randint(30, 150)}"]
    elif primitive_def.construction.name == 'triangle12':
        points = generate_points(0, 3)
        r1 = random.randint(3, 10)
        r2 = random.randint(r1, 10)
        clauses = [f"{' '.join(points)} = triangle12 {' '.join(points)} {r1} {r2}"]
    else:
        points = generate_points(0, len(primitive_def.construction.args))
        clauses = [" ".join(points) + " = " + primitive_def.construction.name]
        
    return clauses, points

def _generate_subsequent_clause(existing_points: list):
    """Generates a subsequent clause by adding a new relation."""
    new_points_num = random.randint(1, 2)
    possible_relation_names = RELATION_SET.get(new_points_num, [])
    if not possible_relation_names:
        return None, None
    
    random.shuffle(possible_relation_names)

    for relation_name in possible_relation_names:
        relation_def = WORKER_DEFINITIONS[relation_name]
        relation_args = relation_def.construction.args
        required_points_count = len(relation_args) - new_points_num
        
        if required_points_count > len(existing_points):
            continue
        
        new_points = generate_points(len(existing_points), new_points_num)
        
        sample_points_count = 3 if relation_def.construction.name == 's_angle' else required_points_count
        if sample_points_count > len(existing_points):
            continue
            
        sample_points = random.sample(existing_points, k=sample_points_count)
        sample_points.sort()
        
        clause_args = sample_points
        if relation_def.construction.name == 's_angle':
            degree = random.randint(30, 150)
            clause_args = sample_points + new_points + [str(degree)]
        
        clause_str = " ".join(new_points) + " = " + relation_def.construction.name + " " + " ".join(clause_args)
        return clause_str, new_points

    return None, None

def generate_sequence(idx, args):
    """The main function for generating a single problem sequence."""
    while True:
        try:
            primitive_name = random.choice(PRIMITIVE_SET)
            clauses, points = _generate_initial_clause(primitive_name)

            for _ in range(args.clause_num):
                new_clause, new_points = _generate_subsequent_clause(points)
                if new_clause:
                    clauses.append(new_clause)
                    points.extend(new_points)
                else:
                    raise ValueError("Failed to generate a subsequent clause.")
            
            full_problem_txt = "; ".join(clauses)

            sequence_seed = hash(full_problem_txt) % (2**32 - 1)
            code_list, image_list, caption_list = [], [], []
            
            for i in range(1, len(clauses) + 1):
                current_code = "; ".join(clauses[:i])
                image_bytes, caption = generate_image_and_caption(
                    current_code, sequence_seed, idx, i, args.image_cache_num, args.image_cache_folder
                )
                if image_bytes is None:
                    raise ValueError("Image generation failed.")
                code_list.append(current_code)
                image_list.append(image_bytes)
                caption_list.append(caption)

            if len(code_list) >= 3:
                base_caption = caption_list[0]
                instruction_list = [
                    caption_list[i].replace(caption_list[i-1], '', 1).strip()
                    for i in range(1, len(caption_list))
                ]
                line_data = {
                    "id": idx, "full_problem_txt": full_problem_txt,
                    "code_list": code_list, "image_list": image_list, 'seed': sequence_seed,
                    "base_caption": base_caption, "instruction_list": instruction_list
                }

                if idx < args.image_cache_num:
                    line_data_no_image = {k: v for k, v in line_data.items() if k not in ['image_list', 'full_problem_txt']}
                    with open(os.path.join(args.image_cache_folder, str(idx), "data.json"), 'w') as f:
                        json.dump(line_data_no_image, f, indent=2)
                
                return line_data
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser(description="Generate foundational geometric structure editing data.")
    parser.add_argument('--dataset_name', type=str, default='foundational_structure_generation', help='Name for the dataset.')
    parser.add_argument('--output_path', type=str, default='./outputs/data', help='Base directory for the final Parquet dataset.')
    parser.add_argument('--image_cache_folder', type=str, default='./outputs/images_editing_cache/', help='Temporary folder to cache a subset of images.')
    parser.add_argument('--defs_path', type=str, default='defs.txt', help='Path to the definitions file.')
    parser.add_argument('--workers_num', type=int, default=max(1, os.cpu_count() - 2), help='Number of worker processes.')
    parser.add_argument('--generation_size', type=int, default=500000, help='Target number of editing sequences to generate.')
    parser.add_argument('--clause_num', type=int, default=3, help='Number of additional clauses (total steps = clause_num + 1).')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for generated data IDs.')
    parser.add_argument('--validation_size', type=int, default=5000, help='Number of examples for the validation set.')
    parser.add_argument('--image_cache_num', type=int, default=1000, help='Number of initial sequences to save image files for debugging.')
    parser.add_argument('--batch_size', type=int, default=100000, help='Number of records per Parquet file batch.')
    args = parser.parse_args()

    definitions = pr.Definition.from_txt_file(args.defs_path, to_dict=True)

    output_path = os.path.join(args.output_path, args.dataset_name)
    os.makedirs(args.image_cache_folder, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    all_results = []
    task_indices = range(args.start_idx, args.start_idx + args.generation_size)
    
    print(f"Starting {args.workers_num} workers to generate up to {args.generation_size} sequences...")
    with multiprocessing.Pool(processes=args.workers_num, initializer=init_worker, initargs=(definitions,)) as pool:
        task_func = partial(generate_sequence, args=args)
        
        with tqdm(total=args.generation_size, desc="Collecting generated sequences") as pbar:
            for result in pool.imap_unordered(task_func, task_indices):
                if result:
                    all_results.append(result)
                    pbar.update(1)

    print(f"\nGeneration complete. Total sequences collected before deduplication: {len(all_results)}")
    if not all_results:
        print("No data was generated. Exiting."); return

    # Deduplicate results after collection
    seen_problems = set()
    unique_results = []
    for result in all_results:
        if result['full_problem_txt'] not in seen_problems:
            seen_problems.add(result['full_problem_txt'])
            del result['full_problem_txt']
            unique_results.append(result)

    print(f"Total unique sequences after deduplication: {len(unique_results)}")

    print("Converting data to Hugging Face Dataset...")
    full_dataset = datasets.Dataset.from_list(unique_results)
  
    print(f"Splitting dataset (validation_size={args.validation_size})...")
    if len(full_dataset) > args.validation_size * 2:
        split_dataset = full_dataset.train_test_split(test_size=args.validation_size, seed=42)
    else:
        print("Dataset too small for a validation split. Using all data for training.")
        split_dataset = datasets.DatasetDict({'train': full_dataset, 'test': datasets.Dataset.from_list([])})

    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Validation set size: {len(split_dataset['test'])}")

    if len(split_dataset['train']) > 0:
        save_dataset_in_batches(split_dataset['train'], 'train', args.dataset_name, output_path, args.batch_size)
    if len(split_dataset['test']) > 0:
        save_dataset_in_batches(split_dataset['test'], 'validation', args.dataset_name, output_path, args.batch_size)
        
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()