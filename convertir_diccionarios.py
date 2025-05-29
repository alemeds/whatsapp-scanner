#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CONVERTIDOR DE DICCIONARIOS TXT A CSV
Convierte automáticamente los diccionarios .txt al formato CSV compatible con Streamlit

Uso:
    python convertir_diccionarios.py
"""

import os
import csv
from pathlib import Path

def convert_txt_to_csv(txt_file, csv_file):
    """Convierte un archivo .txt a .csv manteniendo el formato"""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            
            for line in lines:
                line = line.strip()
                
                # Saltar líneas vacías y comentarios
                if not line or line.startswith('#'):
                    continue
                
                # Procesar líneas con formato término,categoría
                if ',' in line:
                    parts = line.split(',', 1)  # Split solo en la primera coma
                    if len(parts) == 2:
                        termino = parts[0].strip()
                        categoria = parts[1].strip()
                        writer.writerow([termino, categoria])
        
        print(f"✅ Convertido: {txt_file} → {csv_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error convirtiendo {txt_file}: {e}")
        return False

def main():
    """Función principal"""
    print("🔄 Convertidor de Diccionarios TXT a CSV")
    print("=" * 50)
    
    # Buscar archivos .txt en el directorio actual
    txt_files = list(Path('.').glob('diccionario_*.txt'))
    
    if not txt_files:
        print("❌ No se encontraron archivos diccionario_*.txt")
        print("💡 Asegúrate de que los archivos estén en el mismo directorio")
        return
    
    print(f"📁 Encontrados {len(txt_files)} archivos de diccionario:")
    
    converted_count = 0
    
    for txt_file in txt_files:
        # Generar nombre del archivo CSV
        csv_file = txt_file.with_suffix('.csv')
        
        # Mostrar información del archivo
        print(f"\n📄 Procesando: {txt_file.name}")
        
        # Contar líneas del archivo original
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            print(f"   📊 Términos encontrados: {total_lines}")
        except:
            print(f"   ⚠️ No se pudo leer el archivo")
            continue
        
        # Convertir archivo
        if convert_txt_to_csv(txt_file, csv_file):
            converted_count += 1
            
            # Verificar archivo CSV generado
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    csv_lines = sum(1 for row in reader if row)
                print(f"   ✅ CSV generado con {csv_lines} términos")
            except:
                print(f"   ⚠️ No se pudo verificar el CSV generado")
    
    print(f"\n🎉 Conversión completada!")
    print(f"✅ Archivos convertidos: {converted_count}/{len(txt_files)}")
    
    if converted_count > 0:
        print(f"\n📋 Archivos CSV generados:")
        for csv_file in Path('.').glob('diccionario_*.csv'):
            print(f"   • {csv_file.name}")
        
        print(f"\n💡 Ahora puedes usar estos archivos CSV en la aplicación Streamlit")
        print(f"🚀 Para ejecutar la aplicación: streamlit run analizador_whatsapp.py")

if __name__ == "__main__":
    main()