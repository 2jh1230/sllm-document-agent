# -*- coding: utf-8 -*-
"""
AI 엔진의 핵심 로직을 담당하는 AIEngine 클래스를 정의하는 파일입니다.
LLM 로딩, RAG 파이프라인, 구조적 출력 생성 등을 담당합니다.
"""
import os
from typing import List, Dict, Any, Iterator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import config
from utils.text_processing import post_process_response

# --- Pydantic 모델 정의 (구조적 출력용) ---
class FileOperation(BaseModel):
    action: str = Field(description="실행할 작업 유형. 'create_folder' 또는 'move_file' 중 하나여야 합니다.")
    folder_name: str = Field(default=None, description="생성할 폴더의 이름 (action이 'create_folder'일 때 사용).")
    source: str = Field(default=None, description="이동할 원본 파일 또는 폴더의 이름 (action이 'move_file'일 때 사용).")
    destination: str = Field(default=None, description="파일 또는 폴더를 이동할 목적지 경로 (action이 'move_file'일 때 사용).")

class OrganizationPlan(BaseModel):
    plan_description: str = Field(description="제안된 파일 정리 계획에 대한 간략한 설명입니다.")
    commands: List[FileOperation] = Field(description="계획을 실행하기 위한 명령어 목록입니다.")

class PlanSuggestions(BaseModel):
    plans: List[OrganizationPlan] = Field(description="파일 정리를 위한 두 가지 계획 제안 목록입니다.")


class AIEngine:
    def __init__(self):
        self.llm_pipeline = None
        self.tokenizer = None
        self.rag_chain = None
        self._initialize_llm()

    def _initialize_llm(self):
        """8비트 양자화된 언어 모델과 파이프라인을 로드합니다."""
        print("AI 엔진: LLM 로딩을 시작합니다...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=config.MAX_NEW_TOKENS,
            return_full_text=False,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        print("✅ AI 엔진: LLM 로딩 완료!")

    def _get_loader(self, file_path: str):
        """파일 확장자에 맞는 LangChain Document Loader를 반환합니다."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif ext in config.SUPPORTED_EXTENSIONS:
            try:
                return TextLoader(file_path, encoding='utf-8')
            except Exception:
                return TextLoader(file_path, encoding='cp949')
        else:
            return UnstructuredFileLoader(file_path)

    def setup_rag_for_file(self, file_path: str):
        """단일 파일에 대한 RAG 파이프라인을 설정합니다."""
        print(f"AI 엔진: '{file_path}' 파일로 RAG를 설정합니다...")
        try:
            loader = self._get_loader(file_path)
            documents = loader.load()

            if not documents:
                print(f"경고: '{file_path}'에서 문서를 로드하지 못했거나 내용이 비어있습니다.")
                return False

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            texts = text_splitter.split_documents(documents)

            if not texts:
                 print(f"경고: 문서를 청크로 분할하지 못했습니다.")
                 return False
            
            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            vector_db = FAISS.from_documents(texts, embeddings)
            # [수정] as_ retriever -> as_retriever 오타 수정
            retriever = vector_db.as_retriever(search_kwargs={"k": config.SEARCH_K})

            prompt_template = """주어진 문서 내용을 바탕으로 다음 질문에 답변하세요. 문서에 없는 내용은 답변하지 마세요.

문서 내용:
{context}

질문: {question}

답변:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm_pipeline,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
            print(f"✅ AI 엔진: '{os.path.basename(file_path)}' 파일 RAG 준비 완료!")
            return True
        except Exception as e:
            print(f"AI 엔진 오류: RAG 설정 실패 - {e}")
            self.rag_chain = None
            return False

    def run_rag_chat(self, query: str) -> Iterator[str]:
        """설정된 RAG 체인을 사용하여 스트리밍 방식으로 답변을 생성합니다."""
        if not self.rag_chain:
            yield "오류: 먼저 분석할 파일을 선택해야 합니다."
            return

        try:
            result = self.rag_chain.invoke(query)
            response = result.get('result', '답변을 생성하지 못했습니다.')
            processed_response = post_process_response(response)
            
            for char in processed_response:
                yield char
        except Exception as e:
            yield f"채팅 중 오류 발생: {e}"

    def get_organization_suggestion(self, directory: str, file_list: List[str]) -> Dict[str, Any]:
        """
        주어진 파일 목록을 분석하여, Pydantic을 이용한 구조적 출력으로 파일 정리 계획을 제안합니다.
        """
        print(f"AI 엔진: '{directory}' 폴더 정리 계획을 생성합니다...")
        if not file_list:
            return {"error": "폴더가 비어있어 정리할 파일이 없습니다."}

        parser = JsonOutputParser(pydantic_object=PlanSuggestions)

        prompt = PromptTemplate(
            template="""당신은 파일 정리 전문가입니다. 아래 파일 목록을 보고, 사용자가 파일을 쉽게 찾을 수 있도록 두 가지의 서로 다른 정리 계획을 제안하세요. 모든 경로는 상대 경로로 작성해야 합니다.

{format_instructions}

파일 목록:
{file_list}

JSON 응답:""",
            input_variables=["file_list"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm_pipeline | parser

        try:
            file_list_str = "\n".join(file_list)
            response = chain.invoke({"file_list": file_list_str})
            return response
        except Exception as e:
            print(f"AI 엔진 오류: 정리 계획 생성 실패 - {e}")
            return {"error": f"AI가 유효한 정리 계획을 생성하지 못했습니다. 오류: {e}"}