import streamlit as st
from google import genai
from google.genai import types
import requests

# ==========================================
# 1. ตั้งค่าหน้าตา Web App (Frontend Setup)
# ==========================================
st.set_page_config(page_title="Medical AI Pathway Tutor", page_icon="🧬", layout="centered")
st.title("🧬 AI Pathway Tutor")
st.caption("แชทบอทวิเคราะห์ Pathway การแพทย์ระดับลึก (Reactome + Gemini 3.1 Flash Lite)")

# ==========================================
# 2. ตั้งค่า Gemini API Key (แบบปลอดภัยสำหรับขึ้น Cloud)
# ==========================================
# ระบบจะไปดึง Key จากหลังบ้านของ Streamlit Cloud แทนการเขียนฝังไว้ในโค้ด
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("⚠️ ไม่พบ API Key กรุณาตั้งค่าในกล่อง Advanced Settings -> Secrets ก่อน Deploy")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 3. สร้างเครื่องมือ (Tool) สำหรับดึงข้อมูลและภาพ
# ==========================================
def fetch_reactome_pathway(query_term: str) -> dict:
    """ค้นหาและดึงลิงก์ Interactive Browser จาก Reactome"""
    search_url = f"https://reactome.org/ContentService/search/query?query={query_term}"
    
    try:
        res = requests.get(search_url).json()
        if not res.get('results') or not res['results'][0].get('entries'):
            return {"status": "error", "message": "ไม่พบ Pathway นี้ในฐานข้อมูลของ Reactome"}

        st_id = res['results'][0]['entries'][0]['stId']
        name = res['results'][0]['entries'][0]['name']
        
        # ภาพ Preview (แบบภาพนิ่ง)
        diagram_url = f"https://reactome.org/ContentService/exporter/diagram/{st_id}.png"
        
        # ลิงก์สำหรับซูมและคลิกดูรายละเอียด (Interactive)
        interactive_url = f"https://reactome.org/PathwayBrowser/#/{st_id}"

        instruction = (
            f"อธิบาย pathway นี้อย่างละเอียด และแทรกรูปภาพพรีวิวด้วย Markdown: ![{name}]({diagram_url}) \n\n"
            f"**คำสั่งสำคัญ:** ตอนจบคำอธิบาย คุณต้องแนบลิงก์นี้ตัวใหญ่ๆ เพื่อให้ผู้ใช้กดเข้าไปซูมดู: "
            f"[👉 คลิกที่นี่เพื่อเปิดดู {name} แบบ Interactive (ซูมได้/คลิกดูรายละเอียดเอนไซม์ได้)]({interactive_url})"
        )

        return {
            "pathway_name": name,
            "st_id": st_id,
            "instruction_to_ai": instruction
        }
    except Exception as e:
        return {"status": "error", "message": f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}"}

# ==========================================
# 4. ระบบจัดการแชท + ฝัง System Instruction ระดับเชี่ยวชาญ
# ==========================================
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-3.1-flash-lite", 
        config=types.GenerateContentConfig(
            tools=[fetch_reactome_pathway], 
            temperature=0.3, # ลด Temp ให้ตอบเป็นวิชาการและตรงไปตรงมา
            system_instruction=(
                "คุณคือ AI ติวเตอร์แพทย์ระดับเชี่ยวชาญ จงอธิบายเนื้อหาอย่างละเอียดลึกซึ้งระดับ Advanced Medical School "
                "ห้ามสรุปแบบตัดทอนเนื้อหา (Do not omit details) ห้ามข้ามขั้นตอนของกลไกทางชีวเคมีหรือสรีรวิทยา "
                "จงใช้คำศัพท์ทางการแพทย์ที่ถูกต้อง อธิบายตั้งแต่ระดับโมเลกุลไปจนถึง Clinical correlation "
                "หากมีการวิเคราะห์ข้อมูลการวินิจฉัยโรค ให้มุ่งเน้นไปที่ความแม่นยำทางคลินิก (Clinical Accuracy) "
                "และกลไกทางพยาธิสภาพเป็นหลัก โดยไม่ต้องนำประเด็นเรื่องความคุ้มค่าทางเศรษฐศาสตร์ (Cost-effectiveness) มาเกี่ยวข้อง"
            )
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# เรนเดอร์ข้อความประวัติการแชท
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. รับข้อความและแสดงผล
# ==========================================
if user_input := st.chat_input("ลองพิมพ์: อธิบายกระบวนการ Gluconeogenesis แบบเจาะลึก"):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 กำลังค้นหาข้อมูล สกัดกลไก และเตรียม Pathway..."):
            try:
                response = st.session_state.chat_session.send_message(user_input)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"⚠️ พบข้อผิดพลาดในการประมวลผล: {e}")
