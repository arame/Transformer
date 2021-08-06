import pickle, os
from config import Constants, Helper


class Pickle:
    def get_content(filename, get_contents):
        path = os.path.join(Constants.pickle_dir, filename)
        if os.path.exists(path):
            return Pickle.load(filename, path)
        
        return Pickle.save(filename, path, get_contents)

    def save(filename, path, get_contents):
        Helper.printline("Generate tokens")
        output = get_contents()
        Helper.printline(f"Save {filename}")
        with open(path, "wb") as file:
            pickle.dump(output, file)

        Helper.printline(f"{filename} saved")
        return output

    def load(filename, path):
        Helper.printline(f"load {filename}")
        output = None
        with open(path, "rb") as file:
            output = pickle.load(file)

        Helper.printline(f"{filename} loaded")
        return output
        