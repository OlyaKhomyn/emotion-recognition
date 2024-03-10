from real_time import features_model, cnn


def main(model='feat'):
    if model == 'feat':
        features_model()
    else:
        cnn()


if __name__ == "__main__":
    main()
    # main('cnn')
