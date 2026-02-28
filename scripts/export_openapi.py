#!/usr/bin/env python3
"""
Export OpenAPI specification to JSON and YAML files
"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import app

def export_json(output_path: str = "openapi.json"):
    """Export OpenAPI spec as JSON"""
    spec = app.openapi()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"✅ Exported OpenAPI JSON to: {output_path}")

def export_yaml(output_path: str = "openapi.yaml"):
    """Export OpenAPI spec as YAML"""
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML not installed. Installing...")
        os.system("pip install pyyaml -q")
        import yaml
    
    spec = app.openapi()
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"✅ Exported OpenAPI YAML to: {output_path}")

if __name__ == "__main__":
    print("🔄 Exporting OpenAPI Specification...")
    export_json("swagger.json")
    export_yaml("swagger.yaml")
    print("\n✨ Export complete!")
    print("   - swagger.json (OpenAPI 3.x JSON format)")
    print("   - swagger.yaml (OpenAPI 3.x YAML format)")
