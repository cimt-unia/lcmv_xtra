# lcmv_xtra/utils.py

def parse_gpsc(self, filepath):
    """Parse .gpsc file and normalize coordinates to center the origin."""
    channels = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
            channels.append((name, x, y, z))
        except ValueError:
            continue
    return channels
