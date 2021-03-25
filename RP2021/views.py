from datetime import datetime

from django.http import HttpResponse, JsonResponse
from django.template import loader
from data_initialization import get_data
from feature_selection import *
from model_creation import *
import numpy as np
from RP.models import FilterSet, Classifier
import matplotlib.pyplot as plt


def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))


def delete_filter(request):
    id = request.GET['id']
    FilterSet.objects.filter(id=id).delete()
    return JsonResponse({'status': 200})


def save_filter(request):
    filter_set = FilterSet(name=request.GET.get('filters_name', 'unnamed'), info=request.session.get('most_recent_filters', '{}'))
    filter_set.save()
    context = eval(request.session.get('most_recent_filters_report', '{}'))
    template = loader.get_template('filters.html')
    all_saved_filters = FilterSet.objects.all()
    saved_filters = [
        {
            'id': filterSet.id,
            'name': filterSet.name
        } for filterSet in all_saved_filters
    ]
    context['saved_filters'] = saved_filters
    return HttpResponse(template.render(context, request))


def load_filter(request):
    scenario = request.GET["scenario"]
    gadget = request.GET["gadget"]
    instrument = request.GET["instrument"]
    loaded_filter = FilterSet.objects.filter(id=request.GET.get('load'))[0]
    return analyse_filters(eval(loaded_filter.info), request, scenario, gadget, instrument)


def get_filter_info_from_request(request):
    filter_info = {
        'filters': [int(x) for x in request.GET.getlist("methods[]")],
        'filter_sizes': [int(x) for x in request.GET.getlist("quantities[]")],
        'thresholds': [float(x) for x in request.GET.getlist("thresholds[]")]
    }
    filter_info = {
        i: {
            'method': method,
            'n_feat': n_feat,
            'threshold': threshold,
            'name': get_method_name(method)
        } for i, (method, n_feat, threshold) in
        enumerate(zip(filter_info['filters'], filter_info['filter_sizes'], filter_info['thresholds']))
    }
    return filter_info


def analyse_filters(filter_info, request, scenario, gadget, instrument):
    X, y = get_data(scenario, gadget, instrument)
    classes, counts = np.unique(y, return_counts=True)
    y_count = dict(zip(classes, counts))
    post_filter = [
        {
            "title": "Pre-filtering",
            "feat_table": feature_analysis(X, y),
            "corr_table": abs(X.corr()).to_html(classes=["table", "heatmap"], float_format="{:.2f}".format)
        }
    ]

    for i, fil in filter_info.items():
        if fil['method'] == 0:
            X = filter_correlated(X, threshold=fil['threshold'], limit=fil['n_feat'])
            post_filter.append({
                "title": "%d. Remove correlated with threshold %0.2f and %d features limit" % (
                    i + 1, fil['threshold'], fil['n_feat']),
                "n_feat": fil['n_feat'],
                "feat_table": feature_analysis(X, y),
                "corr_table": abs(X.corr()).to_html(classes=["table", "heatmap"], float_format="{:.2f}".format)
            })
        elif fil['method'] < 13:
            X, y, scores = feature_filtering(X, y, n_feat=fil['n_feat'], method=fil['method'])
            post_filter.append({
                "title": "%d. %s => %d features" % (i + 1, get_method_name(fil['method']), fil['n_feat']),
                "scores": scores,
                "score_metric": "metric %d" % fil['method'],
                "n_feat": fil['n_feat'],
                "feat_table": feature_analysis(X, y),
                "corr_table": abs(X.corr()).to_html(classes=["table", "heatmap"], float_format="{:.2f}".format)
            })
        else:
            X, y = feature_reduction(X, y, n_feat=fil['n_feat'], method=fil['method'] - 11)
            post_filter.append({
                "title": "%d. %s => %d features" % (i + 1, get_method_name(fil['method']), fil['n_feat']),
                "n_feat": fil['n_feat'],
                "feat_table": feature_analysis(X, y),
                "corr_table": abs(X.corr()).to_html(classes=["table", "heatmap"], float_format="{:.2f}".format)
            })
        # add plots
        uris = []
        for n_feat in range(1, min(X.shape[1] + 1, 5)):
            uri = plot_features(X, y, interactive=False, n_feat=n_feat)
            uris.append(uri)
        post_filter[-1]["uris"] = uris

    # render
    gadget = "Phone" if gadget.lower() == "phone" else "Watch"
    instrument = "Accelerometer" if instrument.lower() == "accel" else "Gyroscope"
    context = {
        'filters_info': filter_info,
        'scenario': {
            'gadget': gadget,
            'instrument': instrument,
            'n_classes': len(classes),
            'n_feat': len(X.columns),
            'y_count': y_count,
            'n_samples': sum(y_count.values())
        },
        'post_filter': post_filter
    }
    request.session['most_recent_filters'] = str(filter_info)
    request.session['most_recent_filters_report'] = str(context)
    template = loader.get_template('filters.html')
    all_saved_filters = FilterSet.objects.all()
    saved_filters = [
        {
            'id': filterSet.id,
            'name': filterSet.name
        } for filterSet in all_saved_filters
    ]
    context['saved_filters'] = saved_filters
    return HttpResponse(template.render(context, request))


def filters(request):
    if request.GET.get('save', False) != False:
        return save_filter(request)
    elif request.GET.get('load', False) != False:
        return load_filter(request)
    else:
        all_saved_filters = FilterSet.objects.all()
        saved_filters = [
            {
                'id': filterSet.id,
                'name': filterSet.name
            } for filterSet in all_saved_filters
        ]
        try:
            scenario = request.GET["scenario"]
            gadget = request.GET["gadget"]
            instrument = request.GET["instrument"]
            filter_info = get_filter_info_from_request(request)
            return analyse_filters(filter_info, request, scenario, gadget, instrument)
        except:
            context = eval(request.session.get('most_recent_filters_report', '{}'))
            template = loader.get_template('filters.html')
            context['saved_filters'] = saved_filters
            return HttpResponse(template.render(context, request))


def get_filter_by_id(request):
    loaded_filter = FilterSet.objects.filter(id=request.GET.get('load'))[0]
    return eval(loaded_filter.info)


def classifiers(request):
    if request.GET.get('load', False) != False:
        filter_info = get_filter_by_id(request)
        request.session['loaded_class_filters'] = filter_info
    else:
        filter_info = eval(request.session.get('most_recent_filters', '{}'))

    template = loader.get_template('classifiers.html')
    all_saved_filters = FilterSet.objects.all()
    saved_filters = [
        {
            'id': filterSet.id,
            'name': filterSet.name
        } for filterSet in all_saved_filters
    ]
    context = {
        'saved_filters': saved_filters,
        'filters_info': filter_info
    }
    return HttpResponse(template.render(context, request))


def save_classifier(request):
    pipeline_steps = []
    filter_info = request.session.get('loaded_class_filters', {})
    for i, fil in filter_info.items():
        if fil['method'] == 0:
            pipeline_steps.append('FunctionTransformer(filter_correlated, kw_args={\'threshold\': %f, \'limit\':%d})' % (fil['threshold'], fil['n_feat']))
        elif fil['method'] < 12:
            pipeline_steps.append(get_feature_filtering_model(fil['method'], fil['n_feat']))
        else:
            pipeline_steps.append(str(get_feature_reduction_model(fil['method'] - 11, fil['n_feat'])))
    model = request.GET['model']
    name = request.GET['name']
    pipeline_steps.append(model)
    pipeline = 'make_pipeline(' + ', '.join(pipeline_steps) + ')'
    classifier = Classifier(name=name, model=pipeline)
    classifier.save()
    return JsonResponse({"status": 200})


def analyse_classifiers(request):
    scenario = request.GET["scenario"]
    gadget = request.GET["gadget"]
    instrument = request.GET["instrument"]
    X, y = get_data(scenario, gadget, instrument)
    filter_info = request.session.get('loaded_class_filters', {})
    for i, fil in filter_info.items():
        if fil['method'] == 0:
            X = filter_correlated(X, threshold=fil['threshold'], limit=fil['n_feat'])
        elif fil['method'] < 13:
            X, y, _ = feature_filtering(X, y, n_feat=fil['n_feat'], method=fil['method'])
        else:
            X, y = feature_reduction(X, y, n_feat=fil['n_feat'], method=fil['method'] - 11)
    model = request.GET['model']
    validation_splitter = request.GET['validation_method']

    scores, sens_spec, conf_mat = classifier_validation(X, y, model, validation_splitter)
    template = loader.get_template('classifier_report.html')
    context = {
        'classifier': model,
        'time': datetime.now(),
        'scores': scores,
        'sens_spec': sens_spec,
        'conf_mat': confmat_heatmap(conf_mat),
        'img_size': 300 + 50 * len(conf_mat)
    }
    return JsonResponse({"status": 200, "content": template.render(context, request)})


def classifierstacks(request):
    template = loader.get_template('Classifier Stacking.html')
    context = {
        "classifiers": [
            {
                "name": c.name,
                "model": c.model
            } for c in Classifier.objects.all()
        ]
    }
    return HttpResponse(template.render(context, request))


def analyse_stack(request):
    scenario = request.GET["scenario"]
    gadget = request.GET["gadget"]
    instrument = request.GET["instrument"]
    X, y = get_data(scenario, gadget, instrument)

    model = request.GET['model']
    validation_splitter = request.GET['validation_method']

    scores, sens_spec, conf_mat = classifier_validation(X, y, model, validation_splitter)

    template = loader.get_template('classifier_report.html')
    context = {
        'classifier': model,
        'time': datetime.now(),
        'scores': scores,
        'sens_spec': sens_spec,
        'conf_mat': confmat_heatmap(conf_mat),
        'img_size': 300 + 50 * len(conf_mat)
    }
    return JsonResponse({"status": 200, "content": template.render(context, request)})


def help(request):
    template = loader.get_template('help.html')
    return HttpResponse(template.render({}, request))
