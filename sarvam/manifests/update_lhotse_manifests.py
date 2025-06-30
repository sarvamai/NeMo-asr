#!/usr/bin/env python3
import json
import gzip
import os
import glob
from pathlib import Path
import argparse

# Language mapping from canary2.py
lang_map = {
    "marathi": "mr",
    "hindi": "hi", 
    "english": "en",
    "kannada": "kn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "bengali": "bn",
    "gujarati": "gu",
    "odia": "od",
    "punjabi": "pa",
    'assamese': 'od'
}

def process_manifest_file(input_file_path, output_file_path, num_rows=-1):
    """Process a single manifest file and add source_lang/target_lang fields."""
    processed_count = 0
    skipped_count = 0
    
    # create output directory if it doesn't exist
    # output_dir = os.path.dirname(output_file_path)
    # os.makedirs(output_dir, exist_ok=True)
    
    with gzip.open(input_file_path, 'rt', encoding='utf-8') as infile, \
         gzip.open(output_file_path, 'wt', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                cut_data = json.loads(line.strip())
                
                # Check if supervisions exist and have at least one element
                if 'supervisions' not in cut_data or len(cut_data['supervisions']) == 0:
                    print(f"Warning: No supervisions found in line {line_num} of {input_file_path}")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                # Get language from first supervision
                supervision = cut_data['supervisions'][0]
                if 'language' not in supervision:
                    print(f"Warning: No language found in supervision at line {line_num} of {input_file_path}")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                language = supervision['language'].lower()
                
                # Map language to 2-letter code
                if language not in lang_map:
                    print(f"Warning: Unknown language '{language}' at line {line_num} of {input_file_path}")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                lang_code = lang_map[language]
                supervision['language'] = lang_code
                # supervision['lang'] = lang_code
                # supervision['source_lang'] = lang_code
                # supervision['target_lang'] = lang_code
                # supervision['pnc'] = 'yes'
                
                # Ensure custom field exists
                if 'custom' not in supervision:
                    supervision['custom'] = {}
                
                # Add source_lang and target_lang
                supervision['custom']['source_lang'] = lang_code
                supervision['custom']['target_lang'] = lang_code
                supervision['custom']['pnc'] = 'yes'
                supervision['custom']['itn'] = 'yes'
                
                
                # Write modified JSON line
                outfile.write(json.dumps(cut_data, ensure_ascii=False) + '\n')
                processed_count += 1

                if num_rows != -1 and processed_count >= num_rows:
                    break
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num} in {input_file_path}: {e}")
                outfile.write(line)
                skipped_count += 1
            except Exception as e:
                print(f"Unexpected error at line {line_num} in {input_file_path}: {e}")
                outfile.write(line)
                skipped_count += 1
    
    return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description='Update lhotse manifests with source_lang and target_lang fields')
    parser.add_argument('input_dir', help='Input directory containing cuts.*.jsonl.gz files')
    parser.add_argument('output_dir', help='Output directory to save modified manifests')
    parser.add_argument('--pattern', default='cuts.*.jsonl.gz', help='File pattern to match (default: cuts.*.jsonl.gz)')
    # number of rows to ake
    parser.add_argument('--num_rows', type=int, default=-1, help='Number of rows to process (default: all)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all manifest files
    pattern = os.path.join(input_dir, args.pattern)
    manifest_files = glob.glob(pattern)
    
    if not manifest_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(manifest_files)} manifest files to process")
    
    total_processed = 0
    total_skipped = 0
    
    for input_file in sorted(manifest_files):
        input_path = Path(input_file)
        output_path = output_dir / input_path.name
        
        print(f"Processing {input_path.name}...")
        
        try:
            processed, skipped = process_manifest_file(input_path, output_path, args.num_rows)
            total_processed += processed
            total_skipped += skipped
            print(f"  Processed: {processed}, Skipped: {skipped}")
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    print(f"\nTotal processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()