# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tradingagents.llm_adapters import ChatDashScope, ChatDashScopeOpenAI
# 移除 ChatGoogleOpenAI 导入

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_utils import Toolkit

from tradingagents.utils.logging_init import get_logger
logger = get_logger('agents')

from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

class TradingAgentsGraph:
    """交易智能体图的主要编排类"""
    
    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """初始化交易智能体图和组件
        
        Args:
            selected_analysts: 要包含的分析师类型列表
            debug: 是否运行在调试模式
            config: 配置字典，如果为None则使用默认配置
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        
        # 更新接口配置
        set_config(self.config)
        
        # 创建必要的目录
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )
        
        # 初始化LLM
        self._initialize_llms()
        
        # 初始化组件
        self.conditional_logic = ConditionalLogic()
        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)
        
        # 初始化记忆组件
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)
        
        # 初始化工具包和工具节点
        self.toolkit = Toolkit(config=self.config)
        self.tool_nodes = self._initialize_tool_nodes()
        
        # 初始化GraphSetup
        self.setup = GraphSetup(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            toolkit=self.toolkit,
            tool_nodes=self.tool_nodes,
            bull_memory=self.bull_memory,
            bear_memory=self.bear_memory,
            trader_memory=self.trader_memory,
            invest_judge_memory=self.invest_judge_memory,
            risk_manager_memory=self.risk_manager_memory,
            conditional_logic=self.conditional_logic,
            config=self.config
        )
        
        # 构建图
        self.graph = self.setup.setup_graph(selected_analysts)
    
    def _initialize_llms(self):
        """初始化LLM模型"""
        llm_provider = self.config.get("llm_provider", "openai")
        
        if llm_provider == "dashscope":
            self.quick_thinking_llm = ChatDashScope(
                model=self.config.get("quick_think_llm", "qwen-turbo")
            )
            self.deep_thinking_llm = ChatDashScope(
                model=self.config.get("deep_think_llm", "qwen-plus")
            )
        elif llm_provider == "anthropic":
            self.quick_thinking_llm = ChatAnthropic(
                model=self.config.get("quick_think_llm", "claude-3-haiku-20240307")
            )
            self.deep_thinking_llm = ChatAnthropic(
                model=self.config.get("deep_think_llm", "claude-3-sonnet-20240229")
            )
        else:  # 默认使用OpenAI
            self.quick_thinking_llm = ChatOpenAI(
                model=self.config.get("quick_think_llm", "gpt-3.5-turbo"),
                temperature=self.config.get("temperature", 0.7)
            )
            self.deep_thinking_llm = ChatOpenAI(
                model=self.config.get("deep_think_llm", "gpt-4"),
                temperature=self.config.get("temperature", 0.7)
            )
    
    def propagate(self, company_name: str, trade_date: str):
        """执行完整的交易分析流程"""
        # 创建初始状态
        initial_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        
        # 执行图
        graph_args = self.propagator.get_graph_args()
        
        for step in self.graph.stream(initial_state, **graph_args):
            if self.debug:
                print(step)
        
        # 处理最终信号
        final_signal = step.get("final_trade_decision", "")
        decision = self.signal_processor.process_signal(
            final_signal, company_name
        )
        
        return step, decision
    
    def _initialize_tool_nodes(self):
        """初始化工具节点"""
        tool_nodes = {}
        
        # 为每种分析师类型创建工具节点
        analyst_types = ["market", "social", "news", "fundamentals"]
        for analyst_type in analyst_types:
            # 根据分析师类型选择相应的工具
            if analyst_type == "market":
                tools = [self.toolkit.get_stock_market_data_unified]
            elif analyst_type == "fundamentals":
                tools = [self.toolkit.get_stock_fundamentals_unified]
            elif analyst_type == "news":
                tools = [self.toolkit.get_realtime_stock_news]
            elif analyst_type == "social":
                tools = [self.toolkit.get_stock_news_openai]
            else:
                tools = []
            
            tool_nodes[analyst_type] = ToolNode(tools)
        
        return tool_nodes
    
    def reflect_and_remember(self, position_returns: float):
        """反思并记住经验"""
        return self.reflector.reflect_and_remember(position_returns)
    
    def process_signal(self, signal: str, company_name: str):
        """处理交易信号"""
        return self.signal_processor.process_signal(signal, company_name)