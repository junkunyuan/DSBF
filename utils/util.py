from PIL import Image

def load_img(img_pth,):
    """Load images."""
    if img_pth[-1] != "g":
        return Image.open(img_pth+"g")
    return Image.open(img_pth).convert("RGB")

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def write_log(out_file,log_str):
    print(log_str)
    out_file.write(log_str + '\n')
    out_file.flush()

class ForeverDataIterator:
    """A data iterator that will never stop producing data."""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

