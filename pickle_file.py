import pickle, os
from config import Constants, Helper


class Pickle:
    def save(filename, contents):
        Helper.printline(f"Save {filename}")
        path = os.path.join(Constants.pickle_dir, filename)
        if os.path.exists(path):
            Helper.printline(f"Remove previous {filename}")
            os.remove(path)

        with open(path, "wb") as file:
            pickle.dump(contents, file)

        Helper.printline(f"{filename} saved")

    def load(filename):
        path = os.path.join(Constants.pickle_dir, filename)
        if os.path.exists(path) == False:
            return False, None

        Helper.printline(f"load {filename}")
        with open(path, "rb") as file:
            output = pickle.load(file)

        Helper.printline(f"{filename} loaded")
        return True, output
        