from data_initialization import *
from feature_selection import *
from model_creation import *


def load_all_scenarios_for_faster_future_loafing():
    for scenario in ["a", "b", "c"]:
        for gadget in ["phone", "watch"]:
            for instrument in ["accel", "gyro"]:
                df = load_scenario(scenario, gadget, instrument)
                print(df)
    print(get_activities_key())


def main():
    # scenarios = ["a", "b", "c"]
    # gadgets = ["phone", "watch"]
    # instruments = ["accel", "gyro"]
    scenario = "b"
    gadget = "phone"
    instrument = "accel"
    # load data
    X, y = get_data(scenario, gadget, instrument)
    # choose features
    X, y = feature_filtering(X, y, n_feat=3, method=1, threshold=0.9)
    # plot_features(X, y, interactive=False, n_feat=3)
    print(X.columns)
    # plot_correlation(X)
    # do feature reduction
    X, y = feature_reduction(X, y, n_feat=3, method=1)
    # plot_features(X, y, interactive=False)
    # train recognizer
    for i in range(1, 13):
        model, scores = model_creation(X, y, method=i)
        print(model)
        print(scores)
        print("score = %0.3f (+/-) %0.3f" % (np.mean(scores), np.std(scores)))
        print()
    # validate recognizer


if __name__ == "__main__":
    main()
