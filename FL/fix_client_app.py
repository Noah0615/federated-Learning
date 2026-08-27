with open("app/challenge/client_app.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == "from flwr.clientapp import ClientApp":
        continue
    if line.strip() == ",":
        continue
    if "ClientApp" in line and "from flwr.clientapp import ClientApp" not in line and "ClientApp()" not in line:
        continue
    new_lines.append(line)

out = ""
for line in new_lines:
    if line.startswith("from flwr.app import ("):
        out += "from flwr.clientapp import ClientApp\n"
    out += line

with open("app/challenge/client_app.py", "w") as f:
    f.write(out)
