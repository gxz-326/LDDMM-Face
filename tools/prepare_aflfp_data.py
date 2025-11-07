#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to prepare AFLFP dataset in required CSV format.
This script helps convert AFLFP annotations to the format expected by LDDMM-Face.
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image


def get_face_center_and_scale(bbox, img_size=None):
    """
    Calculate face center and scale from bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2] format bounding box
        img_size: (height, width) of the image
    
    Returns:
        center: [center_x, center_y]
        scale: scale factor
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_size = max(x2 - x1, y2 - y1)
    scale = box_size / 200.0
    
    return [center_x, center_y], scale, box_size


def create_aflfp_csv(image_dir, landmarks_data, output_csv, image_list=None):
    """
    Create AFLFP CSV file from landmarks data.
    
    CSV format:
    image_path, scale, box_size, center_x, center_y, x1, y1, x2, y2, ..., x68, y68, palsy_grade
    
    Args:
        image_dir: path to image directory
        landmarks_data: dict with image filenames as keys and values as:
                       {
                           'bbox': [x1, y1, x2, y2],
                           'landmarks': [(x1, y1), (x2, y2), ..., (x68, y68)],  # 68 points
                           'palsy_grade': int (0-6)  # or similar classification
                       }
        output_csv: path to output CSV file
        image_list: optional, list of images to use (if None, uses all in landmarks_data)
    """
    
    if image_list is None:
        image_list = list(landmarks_data.keys())
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        
        for img_name in image_list:
            if img_name not in landmarks_data:
                print(f"Warning: {img_name} not found in landmarks data, skipping...")
                continue
            
            data = landmarks_data[img_name]
            
            # Get center and scale
            center, scale, box_size = get_face_center_and_scale(data['bbox'])
            
            # Prepare row
            row = [img_name, f"{scale:.6f}", f"{box_size:.1f}",
                   f"{center[0]:.1f}", f"{center[1]:.1f}"]
            
            # Add landmarks (68 points, 136 values)
            landmarks = data['landmarks']
            if len(landmarks) != 68:
                print(f"Warning: {img_name} has {len(landmarks)} landmarks instead of 68, skipping...")
                continue
            
            for point in landmarks:
                row.append(f"{point[0]:.1f}")
                row.append(f"{point[1]:.1f}")
            
            # Add palsy grade
            row.append(str(data['palsy_grade']))
            
            writer.writerow(row)
    
    print(f"CSV file created: {output_csv}")


def example_usage():
    """
    Example of how to use this script.
    """
    
    # Example 1: Create from Python dict
    landmarks_data = {
        'image001.jpg': {
            'bbox': [100, 80, 400, 480],  # [x1, y1, x2, y2]
            'landmarks': [
                (150, 100), (160, 105), (170, 110),  # Some random points for illustration
                # ... total of 68 points needed
            ],
            'palsy_grade': 2  # 0-6 scale (0=normal, 1-6=different severity levels)
        },
        'image002.jpg': {
            'bbox': [110, 90, 410, 490],
            'landmarks': [
                # 68 landmark points
            ],
            'palsy_grade': 1
        }
    }
    
    # Create CSV
    # create_aflfp_csv('./images', landmarks_data, './face_landmarks_aflfp_train.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare AFLFP dataset CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a template CSV file
  python tools/prepare_aflfp_data.py --output face_landmarks_aflfp_train.csv --create-template
  
  # Validate CSV format
  python tools/prepare_aflfp_data.py --csv face_landmarks_aflfp_train.csv --validate
        """
    )
    
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--csv', type=str, help='CSV file to validate')
    parser.add_argument('--validate', action='store_true', help='Validate CSV format')
    parser.add_argument('--create-template', action='store_true', help='Create template CSV')
    parser.add_argument('--image-dir', type=str, default='./data/aflfp/images/', 
                        help='Image directory')
    
    return parser.parse_args()


def validate_csv(csv_path):
    """
    Validate AFLFP CSV format.
    
    Expected format:
    image_path, scale, box_size, center_x, center_y, x1, y1, ..., x68, y68, palsy_grade
    Total columns: 1 + 1 + 1 + 2 + 136 + 1 = 142
    """
    
    print(f"Validating CSV: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        
        for idx, row in enumerate(reader, 1):
            # Check column count (should be 142)
            if len(row) != 142:
                print(f"Line {idx}: Error - Expected 142 columns, got {len(row)}")
                continue
            
            # Try to parse numeric fields
            try:
                scale = float(row[1])
                box_size = float(row[2])
                center_x = float(row[3])
                center_y = float(row[4])
                
                # Check landmarks are numeric
                for i in range(5, 141):
                    float(row[i])
                
                palsy_grade = int(row[141])
                
                print(f"Line {idx}: OK (grade={palsy_grade})")
                
            except ValueError as e:
                print(f"Line {idx}: Error parsing numeric values - {e}")
    
    print("Validation complete!")


def create_template_csv(output_path):
    """
    Create a template CSV file for manual annotation.
    """
    
    print(f"Creating template CSV: {output_path}")
    
    # Create header comment
    with open(output_path, 'w') as f:
        f.write("# AFLFP Dataset Format (space-separated)\n")
        f.write("# Columns: image_path scale box_size center_x center_y ")
        f.write("x1 y1 x2 y2 ... x68 y68 palsy_grade\n")
        f.write("# Total: 142 columns (image_path + scale + box_size + center_x + center_y + 136 landmark coords + palsy_grade)\n")
        f.write("# Landmarks: 68 facial points (x, y coordinates for each)\n")
        f.write("# Palsy Grade: 0-6 (House-Brackmann scale or similar)\n")
        f.write("# Example row:\n")
        
        # Create example
        example_row = ['img001.jpg', '1.5', '300', '256', '256']
        # Add 68 random landmark points
        for i in range(68):
            example_row.append(f'{100 + i*2}')
            example_row.append(f'{100 + i}')
        example_row.append('2')
        
        f.write("# " + " ".join(example_row) + "\n")
        f.write("\n")
    
    print(f"Template created at: {output_path}")
    print("Please fill in your data and remove the comment lines.")


if __name__ == '__main__':
    args = parse_args()
    
    if args.create_template and args.output:
        create_template_csv(args.output)
    elif args.validate and args.csv:
        validate_csv(args.csv)
    else:
        print("Please use --help for usage information")
        print("\nQuick start:")
        print("1. Create template: python tools/prepare_aflfp_data.py --output train.csv --create-template")
        print("2. Fill in your data in train.csv")
        print("3. Validate: python tools/prepare_aflfp_data.py --csv train.csv --validate")
