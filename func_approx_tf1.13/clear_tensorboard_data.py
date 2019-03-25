import os, re, shutil

for f in os.listdir("."):
    if re.search("\.tensorboard", f):
        shutil.rmtree(f)
        print("deleted %s" % f)

