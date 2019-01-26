import models
import scipy.io.wavfile as wav
import python_speech_features

opt = DefaultConfig()
viz = Visualizer(opt.env)
assert viz.check_connection()
viz.close()


def audio_to_mfcc(fileurl):
    rate, sig = wav.read(fileurl)
    mfcc_feat = mfcc(sig,rate)
    return mfcc_feat

def train(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)()
    print('====================================================')
    print('CURRENT MODEL:')
    print(model)
    if opt.load_model_path:
        model.load(opt.load_model_path)

    