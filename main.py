####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys
import ssl
import uvicorn
import base64
import io

from fastapi import FastAPI, File, status, Request
from fastapi.responses import RedirectResponse,HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from io import BytesIO

from app import get_image_from_bytes
from app import detect_sample_model
from app import add_bboxs_on_img
from app import get_bytes_from_image
import config, rec_sys

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# def save_openapi_json():
#     '''This function is used to save the OpenAPI documentation 
#     data of the FastAPI application to a JSON file. 
#     The purpose of saving the OpenAPI documentation data is to have 
#     a permanent and offline record of the API specification, 
#     which can be used for documentation purposes or 
#     to generate client libraries. It is not necessarily needed, 
#     but can be helpful in certain scenarios.'''
#     openapi_data = app.openapi()
#     # Change "openapi.json" to desired filename
#     with open("openapi.json", "w") as file:
#         json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


######################### MAIN Func #################################


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)

    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))

    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result

@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_sample_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

def get_bytes_from_image(image):
    # Convert the image object to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    return image_bytes.read()

@app.post("/img_object_detection_to_recipe")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}

    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)

    # Step 3: Predict from model
    predict = detect_sample_model(input_image)
    
    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values
    
    # detected objects to string
    objects_list = detect_res['name'].values.tolist()
    objects_list = ' '.join(set(objects_list))
    # print(len(objects_list))

    if len(objects_list)!=0:
        # recommendations
        recipe = rec_sys.RecSys(objects_list)
        
        response = {}
        count = 0
        for index, row in recipe.iterrows():
            response[count] = {
                'recipe': str(row['recipe']),
                'score': str(row['score']),
                # 'ingredients': str(row['ingredients']),
                # 'Instructions': str(row['Instructions']),
                'Srno': str(row['Srno']).split('.')[0],
                'url': str(row['url'])
            }
            count += 1

        # result json
        result['detect_objects_names'] = ', '.join(objects)
        result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))
        result['recommended_recipes'] = response
    else:
        result['detect_objects'] = 'No ingredients detected'
    
    final_image = add_bboxs_on_img(image = input_image, predict = predict)
    image_bytes = get_bytes_from_image(final_image)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Step 5: Logs and return
    # logger.info("results: {}", result)
    # return result
    return JSONResponse(content={"result": result, "annotated_image": image_base64})
    
# get recipe details using its Srno
@app.get("/recipe/{srno}", response_class=HTMLResponse)
async def get_recipe_by_srno(srno: int, request: Request):
    df = pd.read_csv(config.PARSED_PATH)
    # Check if Srno exists in the DataFrame
    if srno in df["Srno"].values:
        # Filter the DataFrame based on Srno
        recipe_data = df[df["Srno"] == srno].to_dict(orient="records")[0]
        # return recipe_data
        return templates.TemplateResponse("recipe.html", {"request": request, "recipe_data": recipe_data})
    else:
        # return {"error": "Recipe not found"}
        return templates.TemplateResponse("recipe.html", {"request": request, "error": "Recipe not found"})
    
@app.get("/home")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def display_result(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

if __name__ == "__main__":
    import nltk
    nltk.download('wordnet')
    uvicorn.run("main:app", host="0.0.0.0", ssl_keyfile="./certificate/key.pem", ssl_certfile="./certificate/cert.pem", reload=True)