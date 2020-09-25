import json
import logging
from typing import List

import numpy as np
from django.http import HttpResponseRedirect, JsonResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import positional_request, available_images, \
    initialize_env, position_similarity_request, paths_ids
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background
from diplomova_praca_lib.utils import images_with_position_from_json_somhunter
from diplomova_praca_lib.utils import timer
from shared.utils import random_image_path, thumbnail_path, random_subset_image_path, THUMBNAILS_PATH
from .models import PositionRequest, Collage


@csrf_exempt
def alive(request):
    return JsonResponse({"status": "ok"}, status=200)


@csrf_exempt
@timer
def position_similarity_somhunter(request):
    somhunter_request = json.loads(request.POST.get('data', ''))
    collected_images = somhunter_request['collectedImages']

    if not collected_images:
        return JsonResponse({"message": "No images in the collage."}, status=400)

    request = PositionSimilarityRequest(images=images_with_position_from_json_somhunter(collected_images),
                                        source="somhunter")
    response = position_similarity_request(request)

    if 'topn' in somhunter_request:
        # Return also the index of image with distances
        top_count = somhunter_request['topn']

        top_n_distances = response.dissimilarity_scores[:top_count]
        top_n_paths_idxs = paths_ids(response.ranked_paths[:top_count])
        top_n_distances = normalize_distances(top_n_distances)

        return JsonResponse({"idxs": minify_numbers(top_n_paths_idxs), "distances": minify_distances(top_n_distances)},
                            status=200, safe=False)

    # Returns the distances sorted by the image idx, i.e. at second position is the distance of second image.
    sorted_results = np.argsort(response.ranked_paths)
    distances = response.dissimilarity_scores[sorted_results]
    distances = normalize_distances(distances)
    distances = minify_distances(distances)

    return JsonResponse({"distances": distances}, status=200, safe=False)


def minify_numbers(numbers: List[int]):
    return ";".join(str(n) for n in numbers)


def minify_distances(distances: List[float]) -> str:
    return ";".join(str(d) for d in np.around(distances, 10))


def normalize_distances(distances: List[float]) -> List[float]:
    distances = [d / 2. for d in distances]
    return distances


@csrf_exempt
def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    default_method = PositionMethod.REGIONS
    initialize_env('regions')

    subset_images_available = available_images(default_method)

    if not subset_images_available:
        query = random_image_path()
    else:
        query = random_subset_image_path(subset_images_available)

    if query is None:
        return HttpResponseNotFound("Could not load any images.")

    context = {"search_image": query.as_posix()}
    return render(request, 'position_similarity/index.html', context)


@csrf_exempt
def position_similarity_post(request):
    save_request = PositionRequest()
    logging.info("Position similarity request.")

    json_request = json.loads(request.POST['json_data'])
    save_request.json_request = json_request
    images, method, overlay_image = json_request['images'], json_request['method'], json_request['overlay_image']

    request = PositionSimilarityRequest(images=images_with_position_from_json(images),
                                        query_image=path_from_css_background(overlay_image, THUMBNAILS_PATH),
                                        method=PositionMethod.parse(method))
    response = positional_request(request)
    save_request.response = ",".join(response.ranked_paths)

    images_to_render = response.ranked_paths[:100]
    if response.searched_image_rank is not None:
        rank_to_display = response.searched_image_rank + 1
    else:
        rank_to_display = response.searched_image_rank

    context = {
        "ranking_results": [{"img_src": thumbnail_path(path)} for path in images_to_render],
        "search_image_rank": rank_to_display,
    }

    if response.matched_regions:
        context['matched_regions'] = transform_crops_to_rectangles(response.matched_regions, images_to_render)

    save_request.save()
    return JsonResponse(context, status=200)


def transform_crops_to_rectangles(matched_regions, images_to_render):
    return {thumbnail_path(image): list(map(lambda x: x.as_quadruple(), regions)) for image, regions in
            matched_regions.items() if image in images_to_render}


@csrf_exempt
def position_similarity_submit_collage(request):
    json_data = json.loads(request.POST['json_data'])

    collage = Collage()
    collage.overlay_image = json_data['overlay_image']
    collage.images = json_data['images']
    collage.save()

    return JsonResponse({}, status=200)
