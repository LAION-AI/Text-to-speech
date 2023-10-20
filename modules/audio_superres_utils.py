from audiosr import build_model

def load_audiosr(args):
    return build_model(args.model_name, device=args.device)