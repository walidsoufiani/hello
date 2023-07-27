# ====================================================================
# Chargement des librairies
# ===================================================================
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
import requests
import streamlit as st
from PIL import Image, ImageDraw
import layoutparser as lp
import cv2
import numpy as np
from transformers import BertForQuestionAnswering, AutoTokenizer
from transformers import pipeline
import pandas as pd
import json
from pdf2image import convert_from_path
import tempfile
import pytesseract
from streamlit_option_menu import option_menu




# Chemin vers l'exécutable de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



#import PyPDF2


# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header="""
    <head>
        <title>Application Dashboard Crédit Score</title>
        <meta charset="utf-8">
        <meta name="keywords" content="Home Crédit Group, Dashboard, prêt, crédit score">
        <meta name="description" content="Application de Crédit Score - dashboard">
        <meta name="author" content="Loetitia Rabier">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Crimson; font-family:Arial"> AI TOOL FOR FLECHTECK <br>
        <h2 style="color:Gray; font-family:Georgia"> DASHBOARD</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""
# Logo de l'entreprise
logo_path = "logo1.jpeg"
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    st.sidebar.image(logo_image, width=400)
#st.set_page_config(page_title="Prêt à dépenser - Dashboard", page_icon="", layout="wide")
st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)



# ====================================================================
# Upload pdf file
# ====================================================================
def get_default_filename(url):
    # Récupérer la dernière partie du lien comme nom de fichier par défaut
    parts = url.split('/')
    return parts[-1]

def telecharger_fichier_url(url, chemin_destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(chemin_destination, 'wb') as fichier:
            fichier.write(response.content)
        st.success("Téléchargement à partir de l'URL terminé.")
    else:
        st.error("Le téléchargement à partir de l'URL a échoué. Code de statut : {}".format(response.status_code))

def telecharger_fichier_pc(fichier, chemin_destination):
    if fichier is not None:
        with open(chemin_destination, 'wb') as f:
            f.write(fichier.read())
        st.success("Téléchargement à partir de l'ordinateur à réussi.")
    else:
        st.warning("Veuillez sélectionner un fichier.")
# Fonction pour convertir un PDF en images PNG
def convert_pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    images = convert_from_path(temp_pdf_path)
    os.remove(temp_pdf_path)
    return images


# ====================================================================
# Bert Question Answering 
# ====================================================================

#Telechager le model PDF Question answering 
@st.cache_resource
def load_model():
    modelname = 'deepset/bert-base-cased-squad2'
    model = BertForQuestionAnswering.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model2 = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return model2
def load_model2():
    model = pipeline('question-answering')
    return model

# Fonction pour répondre aux questions
def repondre_question(question, context):
    # Votre code pour répondre à la question en utilisant le text
    qa = load_model()
    answer = qa({'question': question,'context': context })
    
    return answer



tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")
# Fonction pour répondre aux questions
def repondre_questions(questions, table):
    reponses = []
    for question in questions:
        if question.strip() != "":
            answer = tqa(table=table.astype(str), query=question)["answer"]
            reponses.append(answer)
        else:
            reponses.append("")
    return reponses  




#def question_answer(file):
    #pdf_file = open(file, "rb")
    #pdf_reader = PyPDF2.PdfReader(pdf_file)
    #text = ""
    #for page_num in range(len(pdf_reader.pages)):
        #page = pdf_reader.pages[page_num]
        #text += page.extract_text()
    #return text
    

    







# 1. as sidebar menu
with st.sidebar:
    sidebar_selection = option_menu(" Menu", ['Home','URL','pdf2_image','Layout parser','Extract Table','Question Answering','footprint','Chatboot','Upload json file','Settings','contact' ], 
        icons=['house','file-earmark-pdf-fill','bi bi-file-earmark-image','bi bi-columns-gap','table','bi bi-patch-question-fill','bi bi-app','bi bi-chat-left-dots-fill','bi bi-filetype-json','gear','bi bi-person-lines-fill'], menu_icon="cast", default_index=1)
    sidebar_selection





if sidebar_selection == 'Home':
    video_file = open('video/FlechTec.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)



if sidebar_selection == 'URL':
    st.title("Télécharger un fichier ")
    # Choix du mode de téléchargement
    choix = st.radio("Choisissez le mode de téléchargement :", ("URL", "Ordinateur"))
    if choix == "URL":
        url = st.text_input("Entrez l'URL du fichier à télécharger :")
        nom_fichier = st.text_input("Entrez le nom du fichier à enregistrer :", value=get_default_filename(url))
        chemin_destination = os.path.join("C:/Users/lenovo/Projet_EBE/uploaded_file", nom_fichier)
        if st.button("Télécharger à partir de l'URL"):
            if url and nom_fichier:
                telecharger_fichier_url(url, chemin_destination)
            else:
                st.warning("Veuillez saisir une URL et un nom de fichier.")
    else:
        fichier = st.file_uploader("Sélectionnez un fichier  :", type="PDF")
        nom_fichier = st.text_input("Entrez le nom du fichier à enregistrer :", value=fichier.name if fichier is not None else "")
        chemin_destination = os.path.join("C:/Users/lenovo/Projet_EBE/uploaded_file", nom_fichier)



if sidebar_selection == 'pdf2_image':
# Titre de l'application
    st.title("Conversion PDF en images")

# Afficher le bouton de téléchargement de fichier PDF
    uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type="pdf")

# Vérifier si un fichier a été téléchargé
    if uploaded_file is not None:
    # Convertir le PDF en images
        images = convert_pdf_to_images(uploaded_file)

    # Afficher les images en rangées sur Streamlit
        row_num = len(images) // 3 + 1
        selected_images = []

    # Afficher le choix pour sélectionner une seule image ou toutes les images
        choice = st.radio("Choisissez une option :", options=["Choisir une  image", "Choisir toutes les images"])

        if choice == "Choisir une  image":
            for row in range(row_num):
                columns = st.columns(3)
                for i, column in enumerate(columns):
                    index = row * 3 + i
                    if index < len(images):
                    # Afficher l'image et la case à cocher correspondante
                        image = images[index]
                        checkbox = column.checkbox(label="", key=f"checkbox_{index}")
                        if checkbox:
                            selected_images.append(image)
                        column.image(image, use_column_width=True)

        else:  # Choisir toutes les images
            selected_images = images

            for row in range(row_num):
                columns = st.columns(3)
                for i, column in enumerate(columns):
                    index = row * 3 + i
                    if index < len(images):
                    # Afficher l'image
                        column.image(images[index], use_column_width=True)

    # Bouton pour sauvegarder les images sélectionnées
        if len(selected_images) > 0:
            st.write("---")
            save_directory = st.text_input("Chemin de sauvegarde des images", value="chemin/vers/repertoire")
            if choice == "Choisir toutes les images":
            
                if st.button("Sauvegarder toutes les images"):
                    for i, image in enumerate(images):
                        image_path = os.path.join(save_directory, f"image_{i}.png")
                        image.save(image_path, "PNG")
                    st.success("Toutes les images sauvegardées avec succès!")
            else:
                if st.button("Sauvegarder les images sélectionnées"):
                    for i, image in enumerate(selected_images):
                        image_path = os.path.join(save_directory, f"image_{i}.png")
                        image.save(image_path, "PNG")
                    st.success("Images sauvegardées avec succès!")








# Chargement du modèle pour layout parser 
config_path = r'C:\Users\lenovo\Downloads\config (12).yaml'
model_path = r'C:\Users\lenovo\Downloads\model_final (6).pth'
model_1 = lp.Detectron2LayoutModel(
    config_path=config_path,
    model_path=model_path,
    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
)



if sidebar_selection == 'Layout parser':
    # Interface utilisateur Streamlit
    st.title("Application pour l'extraction ")
    if st.session_state.get('switch_button', False):
        st.session_state['menu_option'] = (st.session_state.get('menu_option',0) + 1) % 4
        manual_select = st.session_state['menu_option']
    else:
        manual_select = None
    selected4 = option_menu(None, ["Home", "Upload",], 
    icons=['house', 'cloud-upload',], 
    orientation="horizontal", manual_select=manual_select, key='menu_4')
    st.button(f"Move to Next {st.session_state.get('menu_option',1)}", key='switch_button')
    selected4
    option = st.selectbox("Choisissez ce que vous voulez extraire :", ("Texte", "Titre", "Liste", "Figure"))

    # Bouton pour sélectionner le fichier PDF
    uploaded_image = st.file_uploader("Sélectionnez une image", type=["jpg", "jpeg", "png"])


    
    # Vérifier si une image a été téléchargée
    if uploaded_image is not None:
        # Conversion de l'image en format utilisé par LayoutParser
        image = Image.open(uploaded_image).convert("RGB")
        
    
    # Détection  dans l'image
        layout = model_1.detect(image)

        
    # Affichage de l'image avec les tables détectées
        st.image(image, caption="Image d'origine", use_column_width=True)


        if option == "Texte":    
    # Création de l'image de sortie avec les zones de texte détectées
            image_with_text = image.copy()
            draw = ImageDraw.Draw(image_with_text)

    # Liste pour stocker les textes extraits
            textes_extraits = []

    # Parcours des zones de texte détectées
            for i, l in enumerate(layout):
                if l.type == 'Text':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Dessin du cadre autour de la zone de texte dans l'image d'origine
                    draw.rectangle([(x_1, y_1), (x_2, y_2)], outline="blue")

            # Extraction du texte
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    texte = pytesseract.image_to_string(text_image, lang='eng')
                    textes_extraits.append(texte)

    # Affichage de l'image d'origine avec les zones de texte détectées
            st.image(image_with_text, caption="Image d'origine avec les zones de texte détectées", use_column_width=True)






            st.write("Tous les textes extraits ont été enregistrés avec succès.")

    # Création de l'image regroupant toutes les zones de texte
            image_regroupee = Image.new("RGB", image.size)
            for l in layout:
                if l.type == 'Text':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Copie de la zone de texte dans l'image regroupée
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    image_regroupee.paste(text_image, (x_1, y_1))

    # Affichage de l'image regroupant toutes les zones de texte
            st.image(image_regroupee, caption="Image regroupant toutes les zones de texte", use_column_width=True)
        # Enregistrement des textes extraits dans un fichier texte avec un chemin d'accès personnalisé
            save_directory = st.text_input("Chemin de sauvegarde du fichier texte", value="chemin/vers/repertoire")
            save_file_path = os.path.join(save_directory, "textes_extraits.txt")

            if st.button("Enregistrer le texte"):   
                with open(save_file_path, "w", encoding="utf-8") as f:
                    for i, texte in enumerate(textes_extraits):
                        f.write(f"(((Texte {i+1}))) : {texte}\n")
                st.success("Le texte a été enregistré avec succès !")

        elif option == "Titre":
            image_with_text = image.copy()
            draw = ImageDraw.Draw(image_with_text)

    # Liste pour stocker les textes extraits
            textes_extraits = []

    # Parcours des zones de texte détectées
            for i, l in enumerate(layout):
                if l.type == 'Title':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Dessin du cadre autour de la zone de texte dans l'image d'origine
                    draw.rectangle([(x_1, y_1), (x_2, y_2)], outline="blue")

            # Extraction du texte
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    texte = pytesseract.image_to_string(text_image, lang='eng')
                    textes_extraits.append(texte)

    # Affichage de l'image d'origine avec les zones de texte détectées
            st.image(image_with_text, caption="Image d'origine avec les zones de titre détectées", use_column_width=True)






            st.write("Tous les titres extraits ont été enregistrés avec succès.")

    # Création de l'image regroupant toutes les zones de texte
            image_regroupee = Image.new("RGB", image.size)
            for l in layout:
                if l.type == 'Title':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Copie de la zone de texte dans l'image regroupée
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    image_regroupee.paste(text_image, (x_1, y_1))

    # Affichage de l'image regroupant toutes les zones de texte
            st.image(image_regroupee, caption="Image regroupant toutes les zones de titre", use_column_width=True)
        # Enregistrement des textes extraits dans un fichier texte avec un chemin d'accès personnalisé
            save_directory = st.text_input("Chemin de sauvegarde du fichier titre", value="chemin/vers/repertoire")
            save_file_path = os.path.join(save_directory, "titre_extraits.txt")

            if st.button("Enregistrer le titre"):   
                with open(save_file_path, "w", encoding="utf-8") as f:
                    for i, texte in enumerate(textes_extraits):
                        f.write(f"(((Titre {i+1}))) : {texte}\n")
                st.success("Le titre a été enregistré avec succès !")



        elif option == "Liste":
            image_with_text = image.copy()
            draw = ImageDraw.Draw(image_with_text)

    # Liste pour stocker les textes extraits
            textes_extraits = []

    # Parcours des zones de texte détectées
            for i, l in enumerate(layout):
                if l.type == 'List':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Dessin du cadre autour de la zone de texte dans l'image d'origine
                    draw.rectangle([(x_1, y_1), (x_2, y_2)], outline="blue")

            # Extraction du texte
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    texte = pytesseract.image_to_string(text_image, lang='eng')
                    textes_extraits.append(texte)

    # Affichage de l'image d'origine avec les zones de texte détectées
            st.image(image_with_text, caption="Image d'origine avec les zones de liste détectées", use_column_width=True)






            st.write("Tous les listes extraits ont été enregistrés avec succès.")

    # Création de l'image regroupant toutes les zones de texte
            image_regroupee = Image.new("RGB", image.size)
            for l in layout:
                if l.type == 'List':
                    x_1 = int(l.block.x_1)
                    y_1 = int(l.block.y_1)
                    x_2 = int(l.block.x_2)
                    y_2 = int(l.block.y_2)

            # Copie de la zone de texte dans l'image regroupée
                    text_image = image.crop((x_1, y_1, x_2, y_2))
                    image_regroupee.paste(text_image, (x_1, y_1))

    # Affichage de l'image regroupant toutes les zones de texte
            st.image(image_regroupee, caption="Image regroupant toutes les zones de liste", use_column_width=True)
        # Enregistrement des textes extraits dans un fichier texte avec un chemin d'accès personnalisé
            save_directory = st.text_input("Chemin de sauvegarde du fichier liste", value="chemin/vers/repertoire")
            save_file_path = os.path.join(save_directory, "liste_extraits.txt")

            if st.button("Enregistrer la liste"):   
                with open(save_file_path, "w", encoding="utf-8") as f:
                    for i, texte in enumerate(textes_extraits):
                        f.write(f"(((Titre {i+1}))) : {texte}\n")
                st.success("liste a été enregistré avec succès !")




        
        elif option == "Figure":
            image_with_figure = image.copy()
            draw = ImageDraw.Draw(image_with_figure)

    # Création d'une nouvelle image pour afficher les tables détectées (fond noir)
            new_image = Image.new("RGB", image.size, color="black")
            new_draw = ImageDraw.Draw(new_image)

    # Liste pour stocker les coordonnées des tables détectées
            figure_coords_list = []

    # Parcourir les tables détectées
            for i, l in enumerate(layout):
                if l.type == 'Figure':
                    x1 = int(l.block.x_1)
                    y1 = int(l.block.y_1)
                    x2 = int(l.block.x_2)
                    y2 = int(l.block.y_2)

            # Dessiner le cadre de la table en bleu sur l'image avec les cadres bleus
                    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)

            # Extraire l'image de la table
                    figure_image = image.crop((x1, y1, x2, y2))

            # Dessiner l'image de la table dans la nouvelle image (en blanc)
                    new_image.paste(figure_image, (x1, y1))

            # Enregistrer les coordonnées de la table
                    figure_coords_list.append((x1, y1, x2, y2))

    # Afficher l'image avec les cadres bleus autour des tables détectées
            st.image(image_with_figure, caption="Tables détectées (cadres en bleu)", use_column_width=True)

    # Afficher les tables détectées avec leur contenu (fond noir)
            st.image(new_image, caption="Figure détectées (en noir)", use_column_width=True)

    # Bouton pour enregistrer les tables détectées
            save_directory = st.text_input("Chemin du répertoire de destination pour enregistrer les figures", "chemin/vers/le/répertoire")
            save_mode = st.radio("Mode d'enregistrement des figures", ("Enregistrer toutes les figures", "Sélectionner une figures"))

            save_selected_figure = False

            if st.button("Enregistrer les figures"):
                if save_mode == "Enregistrer toutes les figures":
                    for i, figure_coords in enumerate(figure_coords_list):
                        x1, y1, x2, y2 = figure_coords
                        figure_image = image.crop((x1, y1, x2, y2))
                        timestamp = int(time.time())
                        filename = f"figure_{i}_{timestamp}.png"
                        figure_image.save(f"{save_directory}/{filename}")
                st.success("figure enregistrées avec succès !")
            elif save_mode == "Sélectionner une figure ":
                selected_figure = st.selectbox("Sélectionnez figure", range(len(figure_coords_list)))
                save_selected_table = st.button("Enregistrer la figure sélectionnée")

            if save_selected_figure:
                table_coords = figure_coords_list[selected_figure]
                x1, y1, x2, y2 = table_coords
                table_image = image.crop((x1, y1, x2, y2))
                timestamp = int(time.time())
                filename = f"figure_selected_{selected_figure}_{timestamp}.png"
                table_image.save(f"{save_directory}/{filename}")
                st.success("figure sélectionnée enregistrée avec succès !")








# Chargement du modèle
config_path = r'C:\Users\lenovo\Downloads\config (8).yaml'
model_path = r'C:\Users\lenovo\Downloads\model_final (4).pth'
model = lp.Detectron2LayoutModel(
    config_path=config_path,
    model_path=model_path,
    label_map={0:"Table"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
)

if sidebar_selection == 'Extract Table':
    # Titre de l'application
    st.title("Extraction de tables")

# Bouton pour sélectionner l'image
    uploaded_image = st.file_uploader("Sélectionnez une image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
    # Conversion de l'image en format utilisé par LayoutParser
        image = Image.open(uploaded_image).convert("RGB")

    # Détection des tables dans l'image
        layout = model.detect(image)

    # Afficher l'image d'origine
        st.image(image, caption="Image d'origine", use_column_width=True)

    # Création d'une nouvelle image avec les cadres bleus autour des tables détectées
        image_with_tables = image.copy()
        draw = ImageDraw.Draw(image_with_tables)

    # Création d'une nouvelle image pour afficher les tables détectées (fond noir)
        new_image = Image.new("RGB", image.size, color="black")
        new_draw = ImageDraw.Draw(new_image)

    # Liste pour stocker les coordonnées des tables détectées
        table_coords_list = []

    # Parcourir les tables détectées
        for i, l in enumerate(layout):
            if l.type == 'Table':
                x1 = int(l.block.x_1)
                y1 = int(l.block.y_1)
                x2 = int(l.block.x_2)
                y2 = int(l.block.y_2)

            # Dessiner le cadre de la table en bleu sur l'image avec les cadres bleus
                draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)

            # Extraire l'image de la table
                table_image = image.crop((x1, y1, x2, y2))

            # Dessiner l'image de la table dans la nouvelle image (en blanc)
                new_image.paste(table_image, (x1, y1))

            # Enregistrer les coordonnées de la table
                table_coords_list.append((x1, y1, x2, y2))

    # Afficher l'image avec les cadres bleus autour des tables détectées
        st.image(image_with_tables, caption="Tables détectées (cadres en bleu)", use_column_width=True)

    # Afficher les tables détectées avec leur contenu (fond noir)
        st.image(new_image, caption="Tables détectées (en noir)", use_column_width=True)

    # Bouton pour enregistrer les tables détectées
        save_directory = st.text_input("Chemin du répertoire de destination pour enregistrer les tables", "chemin/vers/le/répertoire")
        save_mode = st.radio("Mode d'enregistrement des tables", ("Enregistrer toutes les tables", "Sélectionner une table"))

        save_selected_table = False

        if st.button("Enregistrer les tables"):
            if save_mode == "Enregistrer toutes les tables":
                for i, table_coords in enumerate(table_coords_list):
                    x1, y1, x2, y2 = table_coords
                    table_image = image.crop((x1, y1, x2, y2))
                    timestamp = int(time.time())
                    filename = f"table_{i}_{timestamp}.png"
                    table_image.save(f"{save_directory}/{filename}")
            st.success("Tables enregistrées avec succès !")
        elif save_mode == "Sélectionner une table":
            selected_table = st.selectbox("Sélectionnez une table", range(len(table_coords_list)))
            save_selected_table = st.button("Enregistrer la table sélectionnée")

        if save_selected_table:
            table_coords = table_coords_list[selected_table]
            x1, y1, x2, y2 = table_coords
            table_image = image.crop((x1, y1, x2, y2))
            timestamp = int(time.time())
            filename = f"table_selected_{selected_table}_{timestamp}.png"
            table_image.save(f"{save_directory}/{filename}")
            st.success("Table sélectionnée enregistrée avec succès !")
    



if sidebar_selection == 'Question Answering':
    # Interface utilisateur Streamlit
    st.title("Application Question-Answering")
    if st.session_state.get('switch_button', False):
        st.session_state['menu_option'] = (st.session_state.get('menu_option',0) +1) % 4
        manual_select = st.session_state['menu_option']
    else:
        manual_select = None
    selected4 = option_menu(None, ["texte", "table",],
    icons=['bi bi-file-text', 'bi bi-table',], 
    orientation="horizontal", manual_select=manual_select, key='menu_4')
    
    if selected4 == "texte":
          
    # Bouton pour sélectionner le fichier PDF
        uploaded_file = st.file_uploader("Choisir un fichier Text", type="txt")
    # Vérifier si un fichier a été téléchargé
        if uploaded_file is not None:
            text_reader = pd.read_table( uploaded_file, header=None, delimiter=None)[0].to_list()
            text = ' '.join(map(str,text_reader))
            name_dic = {"name" : 'Name', "type" : "0", "value" : ""}
            answer = repondre_question("what is this document about ?",text)
            name_dic["value"]= answer['answer']
            description_dic = {"name" : 'Description', "type" : "0", "value" : ""}
            answer2 = repondre_question("what does VEML3328 do ?",text)
            description_dic["value"]= "VEML3328 "+answer2['answer']
            dimensions_dic = {"name" : 'Dimensions', "type" : "4", "value" : ""}
            answer3= repondre_question("what are the dimensions ?",text)
            dimensions_dic["value"]= answer3['answer']
            component = {"Parameters " : [name_dic, description_dic, dimensions_dic]}
            json_string = json.dumps(component, indent=4)
            st.json(json_string, expanded=True)
            st.download_button(
                label="Download JSON",
                file_name="Data.json",
                mime="application/json",
                data=json_string,
            )
          
    if selected4 == "table":
    
# Bouton pour sélectionner les fichiers CSV
        upload_files= st.file_uploader("Choisir un ou plusieurs fichiers CSV", type="csv", accept_multiple_files=True)

# Vérifier si des fichiers ont été téléchargés
        if upload_files:
            tables = []  # Liste pour stocker les tables DataFrame Pandas
            for file in upload_files:
                table = pd.read_csv(file)
                tables.append(table)

    # Afficher la liste des tables
            table_names = [file.name for file in upload_files]
            selected_table = st.selectbox("Choisir une table", table_names)

    # Trouver l'indice de la table sélectionnée
            table_index = table_names.index(selected_table)

    # Afficher l'interface utilisateur pour la table sélectionnée
            st.write(f"Table sélectionnée : {selected_table}")
            st.write(tables[table_index])

    # Saisir les questions à poser
            questions = st.text_area("Posez une ou plusieurs questions sur la table, une question par ligne :")

    # Séparer les questions en une liste
            questions = questions.strip().split("\n")

    # Poser les questions et afficher les réponses
            if questions:
                reponses = repondre_questions(questions, tables[table_index])
                for i, reponse in enumerate(reponses):
                    st.write(f"Réponse {i+1} : {reponse}")
            else:
                st.write("Veuillez saisir au moins une question pour continuer.")
        else:
            st.write("Veuillez télécharger un ou plusieurs fichiers CSV pour continuer.")

    



if sidebar_selection == 'footprint':
    st.markdown("⚠️ **Maintenance en cours** ⚠️")
    st.markdown("Notre site est actuellement en cours de maintenance. Veuillez nous excuser pour le dérangement causé. Nous serons de retour sous peu.")


if sidebar_selection == 'Chatboot':
    # Champ de saisie pour la question
    quest = st.text_input("Posez votre question ici :")
    # Bouton pour soumettre la question
    if st.button("Répondre"):
        # Vérifier si une question a été saisie
        if quest:
            # Appeler la fonction pour répondre à la question
            
            answer = repondre_question(quest,text)
            # Afficher la réponse
            st.success(answer['answer'])
        else:
            st.warning("Veuillez saisir une question.")


if sidebar_selection == 'contact':
 
    def st_button(icon, url, label, iconsize):
        if icon == 'youtube':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-youtube" viewBox="0 0 16 16">
                        <path d="M8.051 1.999h.089c.822.003 4.987.033 6.11.335a2.01 2.01 0 0 1 1.415 1.42c.101.38.172.883.22 1.402l.01.104.022.26.008.104c.065.914.073 1.77.074 1.957v.075c-.001.194-.01 1.108-.082 2.06l-.008.105-.009.104c-.05.572-.124 1.14-.235 1.558a2.007 2.007 0 0 1-1.415 1.42c-1.16.312-5.569.334-6.18.335h-.142c-.309 0-1.587-.006-2.927-.052l-.17-.006-.087-.004-.171-.007-.171-.007c-1.11-.049-2.167-.128-2.654-.26a2.007 2.007 0 0 1-1.415-1.419c-.111-.417-.185-.986-.235-1.558L.09 9.82l-.008-.104A31.4 31.4 0 0 1 0 7.68v-.123c.002-.215.01-.958.064-1.778l.007-.103.003-.052.008-.104.022-.26.01-.104c.048-.519.119-1.023.22-1.402a2.007 2.007 0 0 1 1.415-1.42c.487-.13 1.544-.21 2.654-.26l.17-.007.172-.006.086-.003.171-.007A99.788 99.788 0 0 1 7.858 2h.193zM6.4 5.209v4.818l4.157-2.408L6.4 5.209z"/>
                    </svg>  
                    {label}
                </a>
            </p>'''
        elif icon == 'twitter':
            button_code = f'''
            <p>
            <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-twitter" viewBox="0 0 16 16">
                    <path d="M5.026 15c6.038 0 9.341-5.003 9.341-9.334 0-.14 0-.282-.006-.422A6.685 6.685 0 0 0 16 3.542a6.658 6.658 0 0 1-1.889.518 3.301 3.301 0 0 0 1.447-1.817 6.533 6.533 0 0 1-2.087.793A3.286 3.286 0 0 0 7.875 6.03a9.325 9.325 0 0 1-6.767-3.429 3.289 3.289 0 0 0 1.018 4.382A3.323 3.323 0 0 1 .64 6.575v.045a3.288 3.288 0 0 0 2.632 3.218 3.203 3.203 0 0 1-.865.115 3.23 3.23 0 0 1-.614-.057 3.283 3.283 0 0 0 3.067 2.277A6.588 6.588 0 0 1 .78 13.58a6.32 6.32 0 0 1-.78-.045A9.344 9.344 0 0 0 5.026 15z"/>
                </svg>
                {label}
            </a>
            </p>'''
        elif icon == 'linkedin':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>
                    {label}
                </a>
            </p>'''
        elif icon == 'medium':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-medium" viewBox="0 0 16 16">
                        <path d="M9.025 8c0 2.485-2.02 4.5-4.513 4.5A4.506 4.506 0 0 1 0 8c0-2.486 2.02-4.5 4.512-4.5A4.506 4.506 0 0 1 9.025 8zm4.95 0c0 2.34-1.01 4.236-2.256 4.236-1.246 0-2.256-1.897-2.256-4.236 0-2.34 1.01-4.236 2.256-4.236 1.246 0 2.256 1.897 2.256 4.236zM16 8c0 2.096-.355 3.795-.794 3.795-.438 0-.793-1.7-.793-3.795 0-2.096.355-3.795.794-3.795.438 0 .793 1.699.793 3.795z"/>
                    </svg>
                    {label}
                </a>
            </p>'''
        elif icon == 'newsletter':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-envelope" viewBox="0 0 16 16">
                        <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4Zm2-1a1 1 0 0 0-1 1v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2Zm13 2.383-4.708 2.825L15 11.105V5.383Zm-.034 6.876-5.64-3.471L8 9.583l-1.326-.795-5.64 3.47A1 1 0 0 0 2 13h12a1 1 0 0 0 .966-.741ZM1 11.105l4.708-2.897L1 5.383v5.722Z"/>
                    </svg>
                    {label}
                </a>
            </p>'''
        elif icon == 'github':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={iconsize} height={iconsize} fill="currentColor" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    {label}
                </a>
            </p>'''
        elif icon == '':
            button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    {label}
                </a>
            </p>'''
        return st.markdown(button_code, unsafe_allow_html=True)




    st.write("[![Star](https://img.shields.io/github/stars/katanaml/sparrow.svg?logo=github&style=social)](https://github.com/katanaml/sparrow)")

    col1, col2, col3 = st.columns(3)
    col2.image(Image.open('ab.jpg'))

    st.markdown("<h1 style='text-align: center; color: black; font-weight: bold;'>Adrien Girod, Founder FlechTech</h1>",
                        unsafe_allow_html=True)

    st.info('FlècheTech is a Swiss start-up that builds beginner-friendly design tool for prototyping electronic circuit boards, requiring no prior knowledge and appealing to both engineers and hobbyists.')

    icon_size = 20

    st_button('youtube', 'https://www.youtube.com/@AndrejBaranovskij', 'Andrej Baranovskij YouTube channel', icon_size)
    st_button('github', 'https://github.com/katanaml/sparrow', 'Sparrow GitHub', icon_size)
    st_button('twitter', 'https://twitter.com/andrejusb', 'Follow me on Twitter', icon_size)
    st_button('medium', 'https://andrejusb.medium.com', 'Read my Blogs on Medium', icon_size)
    st_button('linkedin', 'https://www.linkedin.com/company/flechetech/', 'Follow me on LinkedIn', icon_size)
    st_button('', 'https://katanaml.io', 'Katana ML', icon_size)











# Cacher le bouton en haut à droite
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

