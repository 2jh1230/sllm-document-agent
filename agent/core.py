# -*- coding: utf-8 -*-
"""
AI 에이전트의 핵심 로직을 담당하는 AIAgent 클래스를 정의하는 파일입니다.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# 설정과 도구를 다른 모듈에서 가져옵니다.
from config import MODEL_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_K, MAX_NEW_TOKENS
from tools.file_system import list_files, move_file, create_folder

class AIAgent:
    def __init__(self):
        self.doc_path = None
        self.selected_file_path = None
        self.agent_executor = None
        self.rag_chain = None
        self.single_file_rag_chain = None
        self.llm_pipeline = None
        self.tokenizer = None

    def _load_llm(self):
        """8비트 양자화된 언어 모델과 파이프라인을 로드합니다."""
        # BitsAndBytesConfig를 사용하여 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # 패딩 토큰 설정 (없다면 eos_token 사용)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 더 강력한 생성 파라미터로 수정
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer, 
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
            do_sample=True,
            temperature=0.05,  # 더 낮은 온도
            top_p=0.8,         # 더 집중된 생성
            repetition_penalty=1.3,  # 더 강한 반복 억제
            no_repeat_ngram_size=3,  # 3-gram 반복 방지
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        print("✅ 8비트 모델 로딩 완료!")

    def _setup_single_file_rag(self, file_path: str):
        """선택된 개별 파일에 대한 RAG 파이프라인을 설정합니다."""
        if not os.path.isfile(file_path):
            print(f"오류: '{file_path}'는 유효한 파일이 아닙니다.")
            self.single_file_rag_chain = None
            return

        print(f"'{file_path}' 파일을 RAG용으로 로딩합니다...")
        
        try:
            # 파일 확장자에 따른 로더 선택
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            else:
                # 텍스트 파일 (.txt, .md, .py, .json 등)
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding='utf-8')
            
            documents = loader.load()
            
            if not documents:
                print(f"경고: '{file_path}' 파일을 로드할 수 없습니다.")
                self.single_file_rag_chain = None
                return
            
            # 단일 파일이므로 더 작은 청크로 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # 더 작은 청크
                chunk_overlap=50
            )
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vector_db = FAISS.from_documents(texts, embeddings)
            
            retriever = vector_db.as_retriever(search_kwargs={"k": 2})  # 더 적은 검색 결과

            # 단일 파일용 간단한 프롬프트
            prompt_template = """다음은 선택된 문서의 내용입니다:

{context}

질문: {question}

위 문서 내용을 바탕으로 간단명료하게 답변하세요:"""
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            self.single_file_rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm_pipeline,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,  # 출처 문서 제외
                chain_type_kwargs={"prompt": prompt}
            )
            
            print(f"✅ '{os.path.basename(file_path)}' 파일 RAG 파이프라인 구축 완료!")
            
        except Exception as e:
            print(f"파일 로딩 오류: {e}")
            self.single_file_rag_chain = None

    def _setup_rag_pipeline(self):
        """선택된 폴더의 문서를 기반으로 RAG 파이프라인을 설정합니다."""
        if not self.doc_path: return

        print(f"\n'{self.doc_path}' 경로의 문서를 RAG용으로 로딩합니다...")
        loader = DirectoryLoader(self.doc_path, show_progress=True, use_multithreading=True)
        documents = loader.load()
        if not documents:
            print(f"경고: '{self.doc_path}'에 문서가 없습니다.")
            self.rag_chain = None
            return
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = FAISS.from_documents(texts, embeddings)
        
        retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K})

        # 훨씬 더 간단하고 직접적인 프롬프트
        prompt_template = """{context}

위 정보를 바탕으로 다음 질문에 답하세요: {question}

답변은 간결하고 명확하게 해주세요."""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm_pipeline,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        print("✅ RAG 채팅 파이프라인 구축 완료!")

    def update_single_file(self, file_path: str):
        """사용자가 개별 파일을 선택하면 호출되어 해당 파일만으로 RAG를 설정합니다."""
        self.selected_file_path = file_path
        self._setup_single_file_rag(file_path)

    def _setup_agent(self):
        """파일 도구와 (존재한다면) RAG 도구를 사용하여 에이전트를 설정합니다."""
        all_tools = [list_files, move_file, create_folder]
        
        if self.rag_chain:
            def run_rag_and_get_source(query: str) -> str:
                result = self.rag_chain.invoke(query)
                answer = result.get('result', '관련 정보를 찾지 못했습니다.')
                sources = [doc.metadata.get('source', '알 수 없는 출처') for doc in result.get('source_documents', [])]
                if sources:
                    unique_sources = list(set(sources))
                    return f"{answer}\n\n[출처: {', '.join(unique_sources)}]"
                return answer

            rag_tool = Tool(
                name="Document_Search",
                func=run_rag_and_get_source,
                description="문서 내용 검색 도구입니다. 선택된 폴더 내 문서에서 정보를 찾을 때 사용하세요."
            )
            all_tools.insert(0, rag_tool)

        # 더 간단하고 명확한 에이전트 프롬프트
        react_prompt_template = """당신은 파일 관리 AI입니다.

사용 가능한 도구: {tools}

형식:
Thought: 무엇을 해야 하는지 생각
Action: {tool_names} 중 선택
Action Input: 도구 입력값
Observation: 결과
Thought: 완료되었는지 확인
Final Answer: 최종 답변

질문: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(react_prompt_template)
        agent = create_react_agent(self.llm_pipeline, all_tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=all_tools, 
            verbose=False,  # 로그 출력 줄이기
            handle_parsing_errors=True,
            max_iterations=5  # 무한 루프 방지
        )
        print("✅ 에이전트 및 도구 설정 완료!")

    def _post_process_response(self, response: str) -> str:
        """응답을 후처리하여 불필요한 태그나 반복을 제거합니다."""
        import re
        
        # 1. XML/HTML 태그 제거
        response = re.sub(r'<[^>]+>', '', response)
        
        # 2. 특수 토큰 및 마크다운 제거
        unwanted_patterns = [
            r'<\|im_start\|>', r'<\|im_end\|>', r'<\|endoftext\|>',
            r'\[INST\]', r'\[/INST\]', r'<s>', r'</s>',
            r'\*\*', r'###', r'assistant', r'user:',
            r'AI:', r'인공지능:', r'답변:', r'응답:'
        ]
        for pattern in unwanted_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # 3. 연속된 특수문자 정리
        response = re.sub(r'[\-=]{3,}', '', response)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        # 4. 문장 중복 제거 (더 정교하게)
        sentences = re.split(r'[.!?]\s*', response)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # 너무 짧은 문장 제외
                # 문장의 핵심 부분만 비교 (공백, 구두점 제거)
                core = re.sub(r'[^\w가-힣]', '', sentence.lower())
                if core not in seen:
                    seen.add(core)
                    unique_sentences.append(sentence)
        
        # 5. 최종 정리
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
            
        return result.strip()

    def initialize(self):
        """AI 에이전트의 모든 구성 요소를 초기화합니다."""
        print("AI 에이전트 초기화를 시작합니다...")
        self._load_llm()
        self._setup_agent()
        print("✅ AI 에이전트 LLM 초기화 완료!")

    def update_doc_path(self, new_path: str):
        """사용자가 새 폴더를 선택하면 호출되어 RAG와 에이전트를 업데이트합니다."""
        self.doc_path = new_path
        self._setup_rag_pipeline()
        self._setup_agent()

    def run_single_file_chat(self, query: str):
        """개별 파일 채팅 모드에서 RAG 체인을 실행합니다."""
        if not self.single_file_rag_chain:
            yield "채팅을 하려면 먼저 파일 탐색기에서 문서를 선택해야 합니다."
            return
        
        try:
            result = self.single_file_rag_chain.invoke(query)
            response = result.get('result', '답변을 생성하지 못했습니다.')
            response = self._post_process_response(response)
            
            # 한 글자씩 스트리밍으로 반환
            for char in response:
                yield char
            
        except Exception as e:
            yield f"단일 파일 채팅 중 오류 발생: {e}"

    def run_chat(self, query: str):
        """채팅 모드에서 RAG 체인을 스트리밍으로 실행합니다."""
        if not self.rag_chain:
            yield "채팅을 하려면 먼저 '폴더 선택'으로 문서가 있는 폴더를 지정해야 합니다."
            return
        
        try:
            # 스트리밍 대신 일반 실행으로 변경하여 안정성 향상
            result = self.rag_chain.invoke(query)
            response = result.get('result', '답변을 생성하지 못했습니다.')
            response = self._post_process_response(response)
            
            # 한 글자씩 스트리밍으로 반환
            for char in response:
                yield char
            
        except Exception as e:
            yield f"채팅 중 오류 발생: {e}"

    def run_agent(self, query: str) -> str:
        """에이전트 모드에서 Agent Executor를 실행합니다."""
        if self.agent_executor:
            try:
                result = self.agent_executor.invoke({"input": query})
                response = result.get('output', "오류: 답변을 생성하지 못했습니다.")
                return self._post_process_response(response)
            except Exception as e:
                return f"에이전트 실행 중 오류 발생: {e}"
        return "에이전트가 아직 초기화되지 않았습니다."


