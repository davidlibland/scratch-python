import os
import json

def build_cluster_q(data, output_dir):
    # Ensure that the director exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key, clustering in data.items():
        path = os.path.join(output_dir, key+".tsv")
        formatted_lines = []
        for label, sublabels in sorted(clustering.items(), key=lambda kv: -len(kv[1])):
            formatted_lines.append(label)
            for sublabel in sublabels:
                formatted_lines.append("\t"+sublabel)
        with open(path, "w") as f:
            f.write("\n".join(formatted_lines))

if __name__ == "__main__":
    f_name = input("what file should we format?")
    while f_name:
        output_dir = f_name
        with open(f_name, "r") as f:
            data = json.load(f)
        processed_f_name = "_"+f_name
        os.rename(f_name, "_"+f_name)
        
        build_cluster_q(data, output_dir)
        f_name=input("what file should we format?")
        