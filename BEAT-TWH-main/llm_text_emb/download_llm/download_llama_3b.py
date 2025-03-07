from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, snapshot_download
from key_token import HUGGINGFACE_TOKEN
import logging
import os

# Configurar logging para mostrar el progreso
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Iniciar sesión en Hugging Face
login(HUGGINGFACE_TOKEN)
logger.info("Sesión iniciada en Hugging Face")

# Directorio donde se guardará el modelo
MODEL_DIR = "./llama-3.2-3b-instruct"
os.makedirs(MODEL_DIR, exist_ok=True)

# Nombre del modelo en Hugging Face
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Descargar los archivos del modelo localmente
logger.info(f"Descargando el modelo desde {model_name}...")
model_dir = snapshot_download(repo_id=model_name, token=HUGGINGFACE_TOKEN, local_dir=MODEL_DIR, resume_download=True)
logger.info(f"Modelo descargado exitosamente en {model_dir}")

# Descargar y cargar el tokenizer
logger.info("Descargando el tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
logger.info("Tokenizer descargado exitosamente")

# Cargar el modelo (sin cuantización)
logger.info("Cargando el modelo...")
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
logger.info("Modelo cargado exitosamente")

# Verificación final
logger.info(f"Descarga completada. Modelos disponibles en {model_dir}")