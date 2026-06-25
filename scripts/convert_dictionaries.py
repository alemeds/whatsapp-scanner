#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert text dictionary files to CSV format."""

import csv
from pathlib import Path


def convert_txt_to_csv(txt_file, csv_file):
    """Convert a .txt file to .csv format."""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        with open(csv_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)

            for line in lines:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        term = parts[0].strip()
                        category = parts[1].strip()
                        writer.writerow([term, category])

        print(f"✅ Converted: {txt_file} → {csv_file}")
        return True

    except OSError as e:
        print(f"❌ Error converting {txt_file}: {e}")
        return False


def main():
    """Main function."""
    print("🔄 Dictionary Converter - TXT to CSV")
    print("=" * 50)

    txt_files = list(Path('.').glob('diccionario_*.txt'))

    if not txt_files:
        print("❌ No diccionario_*.txt files found")
        print("💡 Ensure files are in the same directory")
        return

    print(f"📁 Found {len(txt_files)} dictionary files:")

    converted_count = 0

    for txt_file in txt_files:
        csv_file = txt_file.with_suffix('.csv')

        print(f"\n📄 Processing: {txt_file.name}")

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            print(f"   📊 Terms found: {total_lines}")
        except OSError as e:
            print(f"   ⚠️ Could not read file: {e}")
            continue

        if convert_txt_to_csv(txt_file, csv_file):
            converted_count += 1

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    csv_lines = sum(1 for row in reader if row)
                print(f"   ✅ CSV generated with {csv_lines} terms")
            except OSError as e:
                print(f"   ⚠️ Could not verify CSV: {e}")

    print(f"\n🎉 Conversion complete!")
    print(f"✅ Files converted: {converted_count}/{len(txt_files)}")

    if converted_count > 0:
        print(f"\n📋 Generated CSV files:")
        for csv_file in Path('.').glob('diccionario_*.csv'):
            print(f"   • {csv_file.name}")

        print(f"\n💡 You can now use these CSV files in the Streamlit application")
        print(f"🚀 To run the application: streamlit run whatsapp_analyzer_streamlit.py")


if __name__ == "__main__":
    main()
