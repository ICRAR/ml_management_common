#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2021
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#

import tempfile
from collections import Callable
from typing import TYPE_CHECKING

import aiohttp

from ml_management_common import create_experiment, TaskTypes, BaseExperiment
from aiohttp import web, StreamReader

if TYPE_CHECKING:
    import torch


def run_model_worker(
    preprocess_function: Callable[[str, BaseExperiment], 'torch.Tensor'],
    ml_management_config_file: str,
    model_name: str,
    model_version: str,
    model_worker_port=8080,
    model_worker_host=None
):
    """
    MLFlow model worker that can load a model from MLFlow, then run prediction tasks against the model.

    The model worker provides an HTTP interface for predicting input data:

    - /predict_file: POST request to predict data on the current loaded model.
        request header parameters:
            - Result-Email: If specified, email the result to this email address
            - The input file to predict is the POST request body
        response:
            - 200 OK: If Result-Email is specified, an email will be sent to the address
            - 200 OK: If Result-Email is not specified, the prediction result is returned in the response body
            - 400 Bad Request: If the request body could not be preprocessed correctly
            - 400 Bad Request: If no model is loaded
            - 500 Internal Server Error: If the prediction failed

    - /predict_url: POST request to predict data from a remote URL on the current loaded model.
        request header parameters:
            - Result-Email: If specified, email the result to this email address
        post body parameters:
            - url: The URL to the data to predict
        response:
            - 200 OK: If Result-Email is specified, an email will be sent to the address
            - 200 OK: If Result-Email is not specified, the prediction result is returned in the response body
            - 400 Bad Request: If the request body could not be preprocessed correctly
            - 400 Bad Request: If no model is loaded
            - 500 Internal Server Error: If the prediction failed

    :param preprocess_function Function to call to perform preprocessing on the input data.
    :param ml_management_config_file Path to the ml management configuration file
    :param model_name MLFlow model to load on startup.
    :param model_version MLFlow model version to automatically load on startup, if provided. If not provided, load the latest version.
    :param model_worker_port The port to listen on for HTTP requests
    :param model_worker_host The host to listen on for HTTP requests
    """
    app = web.Application()
    routes = web.RouteTableDef()

    async def download_file(url: str, input_file: str):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                return await write_response(input_file, response.content)

    async def write_response(filename: str, reader: StreamReader):
        with open(filename, "wb") as f:
            async for chunk in reader.iter_chunked(4096):
                f.write(chunk)
            return f.name

    async def predict(input_file: str, output_file: str, exp: BaseExperiment):
        # perform preprocessing
        input_data = preprocess_function(input_file, exp)
        model = exp.download_model(f"models:/{model_name}/{model_version}")
        predictions = model(input_data)
        with open(output_file, "wb") as f:
            f.write(predictions.cpu().numpy().tobytes())
            exp.log_artifact(output_file)

    def response(output_file: str, request: web.Request):
        email = request.headers.get("Result-Email")
        if email is not None:
            # TODO: Send email if requested
            return web.Response(status=200, text=f"Sent email to {email}")
        else:
            # return prediction as file
            with open(output_file, "rb") as f:
                return web.Response(status=200, body=f)

    @routes.post('/predict_file')
    async def predict_file(request: web.Request):
        with create_experiment("model_worker_prediction_file", TaskTypes.application, ml_management_config_file) as exp:
            try:
                if not request.can_read_body:
                    return web.Response(status=400, text="Could not read request body")

                with tempfile.TemporaryDirectory() as tempdir:
                    input_file = f"{tempdir}/input"
                    output_file = f"{tempdir}/output"
                    await write_response(input_file, request.content)
                    await predict(input_file, output_file, exp)
                    return response(output_file, request)
            except Exception as e:
                return web.Response(status=500, text=str(e))

    @routes.post('/predict_url')
    async def predict_url(request: web.Request):
        with create_experiment("model_worker_prediction_url", TaskTypes.application, ml_management_config_file) as exp:
            try:
                if not request.can_read_body:
                    return web.Response(status=400, text="Could not read request body")

                content = await request.json()
                if "url" not in content or isinstance(content["url"], str):
                    return web.Response(status=400, text="Missing 'url' in request body")

                url = content["url"]
                app.logger.info(f"Downloading remote file from {url}")
                with tempfile.TemporaryDirectory() as tempdir:
                    input_file = f"{tempdir}/input"
                    output_file = f"{tempdir}/output"
                    await download_file(url, input_file)
                    await predict(input_file, output_file, exp)
                    return response(output_file, request)
            except Exception as e:
                return web.Response(status=500, text=str(e))

    app.add_routes(routes)
    web.run_app(app, port=model_worker_port, host=model_worker_host)


