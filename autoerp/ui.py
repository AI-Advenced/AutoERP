"""
AutoERP UI Module - User Interface Implementation
Version: 1.0.0
Author: AutoERP Development Team
License: MIT

This module implements the user interface layer for the AutoERP system using Streamlit
for web interface, Plotly for interactive charts, Flask for API endpoints,
and SocketIO for real-time communication.
"""

# ==================== IMPORTS ====================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
import json
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import io
import base64

# Flask and SocketIO for real-time features
from flask import Flask, request, jsonify, session, render_template_string
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
from werkzeug.serving import run_simple

# Core module imports
from autoerp.core import (
    AutoERPApplication, AutoERPConfig, User, BaseModel,
    CRUDService, PaginationInfo, FilterCriteria, ServiceResult,
    UserService, NotificationService, ConnectionManager,
    system_metrics, HealthCheck
)

# Additional UI utilities
import hashlib
import secrets
from functools import wraps
import webbrowser
from urllib.parse import urlencode

# ==================== LOGGING CONFIGURATION ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== STREAMLIT CONFIGURATION ====================

class StreamlitConfig:
    """Streamlit application configuration."""
    
    @staticmethod
    def configure_page():
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AutoERP - Enterprise Resource Planning",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://autoerp.help',
                'Report a bug': 'https://autoerp.bugs',
                'About': "AutoERP v1.0.0 - Hexagonal Architecture ERP System"
            }
        )
        
        # Custom CSS styling
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        .sidebar .sidebar-content {
            padding-top: 1rem;
        }
        
        .dashboard-header {
            background: linear-gradient(90deg, #1f77b4, #17becf);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .status-healthy {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        .data-table {
            border: 1px solid #e0e0e0;
            border-radius: 0.25rem;
        }
        
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)


# ==================== SESSION STATE MANAGEMENT ====================

class SessionState:
    """Streamlit session state management."""
    
    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        
        if 'username' not in st.session_state:
            st.session_state.username = None
        
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'dashboard'
        
        if 'app_instance' not in st.session_state:
            st.session_state.app_instance = None
        
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.now()
        
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
    
    @staticmethod
    def clear():
        """Clear session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    @staticmethod
    def update_activity():
        """Update last activity timestamp."""
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def is_session_expired(timeout_minutes: int = 30) -> bool:
        """Check if session is expired."""
        if 'last_activity' not in st.session_state:
            return True
        
        last_activity = st.session_state.last_activity
        return (datetime.now() - last_activity).total_seconds() > (timeout_minutes * 60)


# ==================== NAVIGATION ====================

class NavigationItem:
    """Navigation menu item."""
    
    def __init__(
        self,
        key: str,
        title: str,
        icon: str = "📄",
        description: str = "",
        requires_auth: bool = True,
        min_role: Optional[str] = None
    ):
        self.key = key
        self.title = title
        self.icon = icon
        self.description = description
        self.requires_auth = requires_auth
        self.min_role = min_role


class NavigationManager:
    """Manages application navigation."""
    
    def __init__(self):
        self.items = [
            NavigationItem("dashboard", "Dashboard", "📊", "System overview and metrics"),
            NavigationItem("users", "Users", "👥", "User management"),
            NavigationItem("data", "Data Management", "🗄️", "Data import and export"),
            NavigationItem("reports", "Reports", "📈", "Business reports and analytics"),
            NavigationItem("notifications", "Notifications", "🔔", "System notifications"),
            NavigationItem("settings", "Settings", "⚙️", "Application settings"),
            NavigationItem("health", "System Health", "❤️", "System health monitoring"),
            NavigationItem("help", "Help", "❓", "Documentation and support", requires_auth=False),
        ]
    
    def render_sidebar(self):
        """Render navigation sidebar."""
        st.sidebar.title("🏢 AutoERP")
        st.sidebar.markdown("---")
        
        # User info
        if st.session_state.authenticated:
            st.sidebar.success(f"Welcome, {st.session_state.username}!")
            st.sidebar.info(f"Role: {st.session_state.user_role}")
            
            if st.sidebar.button("Logout", key="logout_btn"):
                self._handle_logout()
        else:
            st.sidebar.warning("Please login to continue")
        
        st.sidebar.markdown("---")
        
        # Navigation menu
        st.sidebar.subheader("Navigation")
        
        for item in self.items:
            if item.requires_auth and not st.session_state.authenticated:
                continue
            
            # Create button with icon and title
            button_text = f"{item.icon} {item.title}"
            
            if st.sidebar.button(button_text, key=f"nav_{item.key}", help=item.description):
                st.session_state.current_page = item.key
                SessionState.update_activity()
                st.rerun()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**AutoERP v1.0.0**")
        st.sidebar.markdown("Built with Streamlit")
    
    def _handle_logout(self):
        """Handle user logout."""
        SessionState.clear()
        SessionState.initialize()
        st.rerun()
    
    def get_current_page(self) -> str:
        """Get current page from session state."""
        return st.session_state.current_page


# ==================== AUTHENTICATION ====================

class AuthenticationManager:
    """Handles user authentication."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    async def login(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user."""
        try:
            result = await self.app.user_service.authenticate_user(
                username_or_email=username,
                password=password
            )
            
            if result.is_success():
                user, session = result.get_data()
                
                # Update session state
                st.session_state.authenticated = True
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.user_role = user.role.value
                st.session_state.session_token = session.session_token
                SessionState.update_activity()
                
                return True, "Login successful"
            else:
                return False, result.error_message or "Login failed"
        
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False, f"Login error: {str(e)}"
    
    def render_login_form(self):
        """Render login form."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>🏢 AutoERP Login</h1>
            <p>Enterprise Resource Planning System</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.subheader("Please Login")
            
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                login_clicked = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            with col2:
                demo_clicked = st.form_submit_button("Demo Login", use_container_width=True)
            
            if login_clicked and username and password:
                with st.spinner("Authenticating..."):
                    success, message = asyncio.run(self.login(username, password))
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            elif demo_clicked:
                # Demo login with default credentials
                with st.spinner("Logging in as demo user..."):
                    success, message = asyncio.run(self.login("demo", "demo123"))
                
                if success:
                    st.success("Demo login successful")
                    st.rerun()
                else:
                    st.error("Demo login failed. Please check if demo user exists.")
            
            elif login_clicked:
                st.error("Please enter both username and password")
        
        # Additional info
        st.info("💡 **Demo Credentials:** Username: `demo`, Password: `demo123`")
        
        with st.expander("ℹ️ System Information"):
            st.markdown("""
            **AutoERP Features:**
            - 👥 User Management
            - 🗄️ Data Management & Import/Export
            - 📊 Real-time Dashboard
            - 📈 Business Analytics
            - 🔔 Notification System
            - ❤️ Health Monitoring
            
            **Technology Stack:**
            - Backend: Python with Hexagonal Architecture
            - Frontend: Streamlit
            - Database: SQLite/PostgreSQL
            - Charts: Plotly
            - Real-time: SocketIO
            """)


# ==================== DASHBOARD ====================

class DashboardMetrics:
    """Dashboard metrics calculator."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        try:
            # Get health status
            health = await self.app.get_health_status()
            
            # Get application metrics
            app_metrics = self.app.get_metrics()
            
            # Get user statistics (simplified)
            user_repo = self.app.get_repository(User)
            total_users = await user_repo.count()
            active_users = await user_repo.count()  # Simplified - should count active sessions
            
            return {
                'system_status': health.get('status', 'unknown'),
                'uptime_seconds': app_metrics.get('uptime_seconds', 0),
                'total_users': total_users,
                'active_users': active_users,
                'database_status': health.get('components', {}).get('database', {}).get('status', 'unknown'),
                'cache_status': health.get('components', {}).get('cache', {}).get('status', 'unknown'),
                'metrics_count': app_metrics.get('metrics_count', 0)
            }
        
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'system_status': 'error',
                'uptime_seconds': 0,
                'total_users': 0,
                'active_users': 0,
                'database_status': 'error',
                'cache_status': 'error',
                'metrics_count': 0
            }
    
    async def get_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration."""
        # Generate sample business data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)  # For consistent results
        
        data = {
            'date': dates,
            'sales': np.cumsum(np.random.randn(len(dates)) * 10 + 100),
            'expenses': np.cumsum(np.random.randn(len(dates)) * 5 + 50),
            'profit': np.cumsum(np.random.randn(len(dates)) * 8 + 30),
            'users': np.cumsum(np.abs(np.random.randn(len(dates))) * 2 + 1),
            'orders': np.random.poisson(20, len(dates))
        }
        
        df = pd.DataFrame(data)
        df['revenue'] = df['sales'] + np.random.randn(len(df)) * 5
        
        return df


class DashboardPage:
    """Dashboard page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
        self.metrics = DashboardMetrics(app)
    
    def render(self):
        """Render dashboard page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>📊 Dashboard</h1>
            <p>System Overview and Key Performance Indicators</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh", key="dashboard_refresh"):
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", key="auto_refresh")
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Get metrics
        with st.spinner("Loading metrics..."):
            metrics = asyncio.run(self.metrics.get_system_metrics())
        
        # Render metrics cards
        self._render_metrics_cards(metrics)
        
        # Render charts
        self._render_charts()
        
        # Render recent activities
        self._render_recent_activities()
    
    def _render_metrics_cards(self, metrics: Dict[str, Any]):
        """Render metrics cards."""
        st.subheader("📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_class = self._get_status_class(metrics['system_status'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>System Status</h4>
                <p class="{status_class}">{metrics['system_status'].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            uptime_hours = metrics['uptime_seconds'] / 3600
            st.metric(
                label="System Uptime",
                value=f"{uptime_hours:.1f} hours",
                delta=f"{metrics['uptime_seconds']:.0f} seconds"
            )
        
        with col3:
            st.metric(
                label="Total Users",
                value=metrics['total_users'],
                delta=f"{metrics['active_users']} active"
            )
        
        with col4:
            st.metric(
                label="Metrics Collected",
                value=metrics['metrics_count'],
                delta="Real-time"
            )
        
        # Database and Cache status
        col1, col2 = st.columns(2)
        
        with col1:
            db_status_class = self._get_status_class(metrics['database_status'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🗄️ Database</h4>
                <p class="{db_status_class}">{metrics['database_status'].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cache_status_class = self._get_status_class(metrics['cache_status'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>💾 Cache</h4>
                <p class="{cache_status_class}">{metrics['cache_status'].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _get_status_class(self, status: str) -> str:
        """Get CSS class for status."""
        if status in ['healthy', 'success']:
            return 'status-healthy'
        elif status in ['warning', 'degraded']:
            return 'status-warning'
        else:
            return 'status-error'
    
    def _render_charts(self):
        """Render dashboard charts."""
        st.subheader("📊 Analytics")
        
        # Get sample data
        with st.spinner("Loading chart data..."):
            df = asyncio.run(self.metrics.get_sample_data())
        
        # Create tabs for different chart types
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Performance", "👥 Users", "💰 Financial"])
        
        with tab1:
            self._render_trend_charts(df)
        
        with tab2:
            self._render_performance_charts(df)
        
        with tab3:
            self._render_user_charts(df)
        
        with tab4:
            self._render_financial_charts(df)
    
    def _render_trend_charts(self, df: pd.DataFrame):
        """Render trend charts."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Sales and Revenue Trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sales'],
            mode='lines',
            name='Sales',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title="Sales and Revenue Trends",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Orders Over Time
        fig2 = px.line(
            df, 
            x='date', 
            y='orders',
            title='Daily Orders',
            labels={'orders': 'Number of Orders', 'date': 'Date'}
        )
        fig2.update_traces(line_color='#2ca02c')
        fig2.update_layout(height=300)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_performance_charts(self, df: pd.DataFrame):
        """Render performance charts."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Performance Metrics Gauge
        col1, col2 = st.columns(2)
        
        with col1:
            # System Performance Gauge
            performance_score = np.random.uniform(75, 95)  # Simulated performance score
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=performance_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Performance"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response Time Chart
            response_times = np.random.exponential(2, 100)  # Simulated response times
            
            fig = go.Figure(data=[go.Histogram(x=response_times, nbinsx=20)])
            fig.update_layout(
                title="Response Time Distribution",
                xaxis_title="Response Time (seconds)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Resource Utilization
        categories = ['CPU', 'Memory', 'Disk', 'Network']
        values = np.random.uniform(20, 80, 4)  # Simulated utilization
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ])
        fig.update_layout(
            title="Resource Utilization (%)",
            yaxis=dict(range=[0, 100]),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_user_charts(self, df: pd.DataFrame):
        """Render user-related charts."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User Growth Chart
            fig = px.line(
                df,
                x='date',
                y='users',
                title='User Growth Over Time',
                labels={'users': 'Total Users', 'date': 'Date'}
            )
            fig.update_traces(line_color='#9467bd')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # User Activity Heatmap (Simulated)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            hours = list(range(24))
            
            # Generate simulated activity data
            np.random.seed(42)
            activity_data = np.random.poisson(10, (7, 24))
            
            fig = go.Figure(data=go.Heatmap(
                z=activity_data,
                x=hours,
                y=days,
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="User Activity Heatmap",
                xaxis_title="Hour of Day",
                yaxis_title="Day of Week",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # User Role Distribution (Pie Chart)
        roles = ['Admin', 'Manager', 'User', 'Guest']
        role_counts = np.random.multinomial(100, [0.1, 0.2, 0.6, 0.1])
        
        fig = px.pie(
            values=role_counts,
            names=roles,
            title="User Role Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_financial_charts(self, df: pd.DataFrame):
        """Render financial charts."""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Profit and Loss Chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profit Over Time', 'Revenue vs Expenses', 'Monthly Summary', 'Profit Margin'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Profit Over Time
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['profit'], name='Profit', line=dict(color='green')),
            row=1, col=1
        )
        
        # Revenue vs Expenses
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['revenue'], name='Revenue', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['expenses'], name='Expenses', line=dict(color='red')),
            row=1, col=2
        )
        
        # Monthly Summary (Bar Chart)
        monthly_data = df.groupby(df['date'].dt.month).agg({
            'sales': 'sum',
            'expenses': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(x=monthly_data['date'], y=monthly_data['sales'], name='Monthly Sales'),
            row=2, col=1
        )
        
        # Profit Margin
        df['profit_margin'] = (df['profit'] / df['revenue']) * 100
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['profit_margin'], name='Profit Margin %', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Financial Dashboard",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df['revenue'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}", delta="YTD")
        
        with col2:
            total_profit = df['profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.0f}", delta="YTD")
        
        with col3:
            avg_profit_margin = df['profit_margin'].mean()
            st.metric("Avg Profit Margin", f"{avg_profit_margin:.1f}%", delta="Monthly")
        
        with col4:
            total_expenses = df['expenses'].sum()
            st.metric("Total Expenses", f"${total_expenses:,.0f}", delta="YTD")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_recent_activities(self):
        """Render recent activities section."""
        st.subheader("📝 Recent Activities")
        
        # Simulated recent activities
        activities = [
            {"time": "2 minutes ago", "user": "admin", "action": "Created new user account", "status": "success"},
            {"time": "5 minutes ago", "user": "john_doe", "action": "Updated profile information", "status": "info"},
            {"time": "10 minutes ago", "user": "system", "action": "Automated backup completed", "status": "success"},
            {"time": "15 minutes ago", "user": "jane_smith", "action": "Exported financial report", "status": "info"},
            {"time": "20 minutes ago", "user": "admin", "action": "System maintenance scheduled", "status": "warning"},
            {"time": "25 minutes ago", "user": "bob_wilson", "action": "Failed login attempt", "status": "error"},
        ]
        
        # Create activities dataframe
        activities_df = pd.DataFrame(activities)
        
        # Style the dataframe
        def style_status(val):
            colors = {
                'success': 'background-color: #d4edda; color: #155724',
                'info': 'background-color: #d1ecf1; color: #0c5460',
                'warning': 'background-color: #fff3cd; color: #856404',
                'error': 'background-color: #f8d7da; color: #721c24'
            }
            return colors.get(val, '')
        
        styled_df = activities_df.style.applymap(style_status, subset=['status'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "time": "Time",
                "user": "User",
                "action": "Action",
                "status": "Status"
            }
        )
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👥 Add User", use_container_width=True):
                st.session_state.current_page = 'users'
                st.rerun()
        
        with col2:
            if st.button("📊 View Reports", use_container_width=True):
                st.session_state.current_page = 'reports'
                st.rerun()
        
        with col3:
            if st.button("🗄️ Import Data", use_container_width=True):
                st.session_state.current_page = 'data'
                st.rerun()
        
        with col4:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.current_page = 'settings'
                st.rerun()


# ==================== DATA MANAGEMENT ====================

class DataManagementPage:
    """Data management page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render data management page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>🗄️ Data Management</h1>
            <p>Import, Export, and Manage System Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📥 Import Data", "📤 Export Data", "🔍 Browse Data"])
        
        with tab1:
            self._render_import_section()
        
        with tab2:
            self._render_export_section()
        
        with tab3:
            self._render_browse_section()
    
    def _render_import_section(self):
        """Render data import section."""
        st.subheader("Import Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file to import",
            type=['csv', 'json', 'xlsx'],
            help="Supported formats: CSV, JSON, Excel"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes",
                "File type": uploaded_file.type
            }
            
            st.json(file_details)
            
            # Preview data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Import configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    table_name = st.text_input("Target Table Name", value="imported_data")
                    batch_size = st.number_input("Batch Size", value=1000, min_value=100, max_value=10000)
                
                with col2:
                    on_conflict = st.selectbox("On Conflict", ["ignore", "update", "error"])
                    validate_data = st.checkbox("Validate Data Before Import", value=True)
                
                if st.button("Import Data", type="primary"):
                    with st.spinner("Importing data..."):
                        # Convert DataFrame to list of dicts
                        data_records = df.to_dict('records')
                        
                        # Simulate import process
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(len(data_records)):
                            progress = (i + 1) / len(data_records)
                            progress_bar.progress(progress)
                            status_text.text(f'Processing record {i + 1} of {len(data_records)}')
                            time.sleep(0.01)  # Simulate processing time
                        
                        st.success(f"Successfully imported {len(data_records)} records to table '{table_name}'")
                        
                        # Show import summary
                        st.json({
                            "Records imported": len(data_records),
                            "Target table": table_name,
                            "Batch size": batch_size,
                            "Conflict handling": on_conflict
                        })
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def _render_export_section(self):
        """Render data export section."""
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Table selection
            available_tables = ["users", "notifications", "audit_logs", "sessions"]  # Simulated
            selected_table = st.selectbox("Select Table", available_tables)
            
            # Date range
            date_from = st.date_input("From Date", value=date.today() - timedelta(days=30))
            date_to = st.date_input("To Date", value=date.today())
        
        with col2:
            # Export format
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
            
            # Additional options
            include_headers = st.checkbox("Include Headers", value=True)
            compress_file = st.checkbox("Compress Output", value=False)
        
        # Filters
        st.subheader("Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_field = st.selectbox("Filter Field", ["All", "id", "created_at", "status"])
        
        with col2:
            filter_operator = st.selectbox("Operator", ["equals", "contains", "greater_than", "less_than"])
        
        with col3:
            filter_value = st.text_input("Filter Value")
        
        if st.button("Export Data", type="primary"):
            with st.spinner("Preparing export..."):
                # Simulate export process
                sample_data = {
                    'users': pd.DataFrame({
                        'id': range(1, 101),
                        'username': [f'user_{i}' for i in range(1, 101)],
                        'email': [f'user_{i}@example.com' for i in range(1, 101)],
                        'created_at': pd.date_range(start='2024-01-01', periods=100, freq='D')
                    }),
                    'notifications': pd.DataFrame({
                        'id': range(1, 51),
                        'subject': [f'Notification {i}' for i in range(1, 51)],
                        'status': np.random.choice(['sent', 'pending', 'failed'], 50),
                        'created_at': pd.date_range(start='2024-01-01', periods=50, freq='2D')
                    })
                }
                
                df = sample_data.get(selected_table, pd.DataFrame())
                
                if not df.empty:
                    # Apply date filter
                    if 'created_at' in df.columns:
                        df = df[
                            (df['created_at'].dt.date >= date_from) &
                            (df['created_at'].dt.date <= date_to)
                        ]
                    
                    # Apply custom filter
                    if filter_field != "All" and filter_value:
                        if filter_operator == "equals":
                            df = df[df[filter_field].astype(str) == filter_value]
                        elif filter_operator == "contains":
                            df = df[df[filter_field].astype(str).str.contains(filter_value, na=False)]
                    
                    # Prepare download
                    if export_format == "CSV":
                        output = df.to_csv(index=False)
                        file_extension = "csv"
                        mime_type = "text/csv"
                    elif export_format == "JSON":
                        output = df.to_json(orient='records', date_format='iso')
                        file_extension = "json"
                        mime_type = "application/json"
                    else:  # Excel
                        output_buffer = io.BytesIO()
                        df.to_excel(output_buffer, index=False)
                        output = output_buffer.getvalue()
                        file_extension = "xlsx"
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    
                    # Create download button
                    filename = f"{selected_table}_export_{date.today().strftime('%Y%m%d')}.{file_extension}"
                    
                    st.download_button(
                        label=f"Download {export_format} File",
                        data=output,
                        file_name=filename,
                        mime=mime_type
                    )
                    
                    st.success(f"Export ready! {len(df)} records prepared for download.")
                    
                    # Show preview
                    st.subheader("Export Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                
                else:
                    st.warning("No data found for the selected criteria.")
    
    def _render_browse_section(self):
        """Render data browsing section."""
        st.subheader("Browse System Data")
        
        # Table selector
        tables = ["users", "notifications", "audit_logs", "sessions", "permissions"]
        selected_table = st.selectbox("Select Table to Browse", tables, key="browse_table")
        
        # Create sample data for demonstration
        if selected_table == "users":
            data = self._get_sample_users_data()
        elif selected_table == "notifications":
            data = self._get_sample_notifications_data()
        elif selected_table == "audit_logs":
            data = self._get_sample_audit_data()
        else:
            data = pd.DataFrame({"message": ["No data available for this table"]})
        
        # Search and filter
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search", placeholder="Enter search term...")
        
        with col2:
            records_per_page = st.selectbox("Records per page", [10, 25, 50, 100], index=1)
        
        with col3:
            show_filters = st.checkbox("Show Advanced Filters")
        
        if show_filters:
            st.subheader("Advanced Filters")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                if selected_table == "users":
                    role_filter = st.multiselect("Role", ["admin", "user", "manager"], key="role_filter")
                    status_filter = st.selectbox("Status", ["All", "Active", "Inactive"], key="status_filter")
            
            with filter_col2:
                date_range = st.date_input(
                    "Created Date Range",
                    value=[date.today() - timedelta(days=30), date.today()],
                    key="date_range_filter"
                )
        
        # Apply search filter
        if search_term and isinstance(data, pd.DataFrame) and not data.empty:
            mask = data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            data = data[mask]
        
        # Pagination
        if isinstance(data, pd.DataFrame) and not data.empty:
            total_records = len(data)
            total_pages = (total_records + records_per_page - 1) // records_per_page
            
            # Page selector
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                current_page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    key="current_page",
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            # Calculate pagination
            start_idx = (current_page - 1) * records_per_page
            end_idx = start_idx + records_per_page
            page_data = data.iloc[start_idx:end_idx]
            
            # Display info
            st.info(f"Showing records {start_idx + 1}-{min(end_idx, total_records)} of {total_records}")
            
            # Display data
            st.dataframe(
                page_data,
                use_container_width=True,
                hide_index=True,
                column_config=self._get_column_config(selected_table)
            )
            
            # Bulk actions
            if selected_table in ["users", "notifications"]:
                st.subheader("Bulk Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📧 Send Notification"):
                        st.info("Bulk notification feature coming soon!")
                
                with col2:
                    if st.button("🗑️ Delete Selected"):
                        st.warning("Bulk delete feature coming soon!")
                
                with col3:
                    if st.button("📤 Export Selection"):
                        st.info("Export selection feature coming soon!")
        
        else:
            st.warning("No data found matching the criteria.")
    
    def _get_sample_users_data(self) -> pd.DataFrame:
        """Get sample users data."""
        return pd.DataFrame({
            'id': range(1, 26),
            'username': [f'user_{i:02d}' for i in range(1, 26)],
            'email': [f'user_{i:02d}@example.com' for i in range(1, 26)],
            'role': np.random.choice(['admin', 'user', 'manager'], 25),
            'status': np.random.choice(['Active', 'Inactive'], 25, p=[0.8, 0.2]),
            'created_at': pd.date_range(start='2024-01-01', periods=25, freq='5D'),
            'last_login': pd.date_range(start='2024-11-01', periods=25, freq='1D')
        })
    
    def _get_sample_notifications_data(self) -> pd.DataFrame:
        """Get sample notifications data."""
        return pd.DataFrame({
            'id': range(1, 21),
            'subject': [f'Notification Subject {i}' for i in range(1, 21)],
            'recipient': [f'user_{i:02d}@example.com' for i in range(1, 21)],
            'status': np.random.choice(['sent', 'pending', 'failed'], 20),
            'channel': np.random.choice(['email', 'sms', 'in_app'], 20),
            'created_at': pd.date_range(start='2024-12-01', periods=20, freq='6H'),
            'sent_at': pd.date_range(start='2024-12-01', periods=20, freq='6H') + pd.Timedelta(minutes=5)
        })
    
    def _get_sample_audit_data(self) -> pd.DataFrame:
        """Get sample audit data."""
        actions = ['CREATE_USER', 'UPDATE_USER', 'DELETE_USER', 'LOGIN', 'LOGOUT', 'EXPORT_DATA']
        return pd.DataFrame({
            'id': range(1, 31),
            'user_id': [f'user_{np.random.randint(1, 10):02d}' for _ in range(30)],
            'action': np.random.choice(actions, 30),
            'resource_type': np.random.choice(['User', 'Notification', 'Report'], 30),
            'success': np.random.choice([True, False], 30, p=[0.9, 0.1]),
            'ip_address': [f'192.168.1.{np.random.randint(1, 255)}' for _ in range(30)],
            'created_at': pd.date_range(start='2024-12-01', periods=30, freq='2H')
        })
    
    def _get_column_config(self, table_name: str) -> Dict[str, Any]:
        """Get column configuration for different tables."""
        configs = {
            'users': {
                'status': st.column_config.SelectboxColumn(
                    'Status',
                    options=['Active', 'Inactive'],
                    width='small'
                ),
                'created_at': st.column_config.DatetimeColumn(
                    'Created At',
                    format='DD/MM/YYYY HH:mm'
                ),
                'last_login': st.column_config.DatetimeColumn(
                    'Last Login',
                    format='DD/MM/YYYY HH:mm'
                )
            },
            'notifications': {
                'status': st.column_config.SelectboxColumn(
                    'Status',
                    options=['sent', 'pending', 'failed'],
                    width='small'
                ),
                'channel': st.column_config.SelectboxColumn(
                    'Channel',
                    options=['email', 'sms', 'in_app'],
                    width='small'
                )
            }
        }
        
        return configs.get(table_name, {})

# ==================== USER MANAGEMENT ====================

class UserManagementPage:
    """User management page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render user management page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>👥 User Management</h1>
            <p>Manage system users, roles, and permissions</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Users", "🔒 Roles", "🛡️ Permissions", "📊 Analytics"])
        
        with tab1:
            self._render_users_section()
        
        with tab2:
            self._render_roles_section()
        
        with tab3:
            self._render_permissions_section()
        
        with tab4:
            self._render_user_analytics()
    
    def _render_users_section(self):
        """Render users management section."""
        # Action buttons
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.subheader("System Users")
        
        with col2:
            if st.button("➕ Add User", key="add_user_btn"):
                st.session_state.show_add_user_modal = True
        
        with col3:
            if st.button("📤 Export Users", key="export_users_btn"):
                self._export_users()
        
        with col4:
            if st.button("🔄 Refresh", key="refresh_users_btn"):
                st.rerun()
        
        # User creation modal
        if st.session_state.get('show_add_user_modal', False):
            self._render_add_user_modal()
        
        # Search and filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_user = st.text_input("🔍 Search Users", placeholder="Username, email, or name...")
        
        with col2:
            role_filter = st.selectbox("Filter by Role", ["All", "Admin", "Manager", "User", "Guest"])
        
        with col3:
            status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive", "Locked"])
        
        # Get users data
        users_data = self._get_users_data(search_user, role_filter, status_filter)
        
        # Display users table
        if not users_data.empty:
            # Pagination
            records_per_page = 10
            total_records = len(users_data)
            total_pages = (total_records + records_per_page - 1) // records_per_page
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    key="users_page",
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            
            start_idx = (page - 1) * records_per_page
            end_idx = start_idx + records_per_page
            page_data = users_data.iloc[start_idx:end_idx]
            
            # Enhanced user table with actions
            st.markdown("### User List")
            
            for idx, user in page_data.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 2])
                    
                    with col1:
                        st.write(f"**{user['username']}**")
                        st.caption(user['email'])
                    
                    with col2:
                        st.write(user['full_name'])
                        st.caption(f"ID: {user['id']}")
                    
                    with col3:
                        role_color = {
                            'admin': '🔴',
                            'manager': '🟡', 
                            'user': '🟢',
                            'guest': '🔵'
                        }
                        st.write(f"{role_color.get(user['role'], '⚪')} {user['role'].title()}")
                    
                    with col4:
                        status_icon = "✅" if user['status'] == 'Active' else "❌"
                        st.write(f"{status_icon} {user['status']}")
                    
                    with col5:
                        st.caption(f"Login: {user['last_login']}")
                    
                    with col6:
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            if st.button("✏️", key=f"edit_{user['id']}", help="Edit user"):
                                self._edit_user(user['id'])
                        
                        with action_col2:
                            if st.button("🔒" if user['status'] == 'Active' else "🔓", 
                                       key=f"lock_{user['id']}", 
                                       help="Lock/Unlock user"):
                                self._toggle_user_lock(user['id'])
                        
                        with action_col3:
                            if st.button("🗑️", key=f"delete_{user['id']}", help="Delete user"):
                                self._delete_user(user['id'])
                    
                    st.divider()
            
            st.info(f"Showing {start_idx + 1}-{min(end_idx, total_records)} of {total_records} users")
        
        else:
            st.warning("No users found matching the search criteria.")
    
    def _render_add_user_modal(self):
        """Render add user modal."""
        st.markdown("### Add New User")
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username *", placeholder="Enter username")
                first_name = st.text_input("First Name *", placeholder="Enter first name")
                role = st.selectbox("Role *", ["user", "manager", "admin"])
            
            with col2:
                email = st.text_input("Email *", placeholder="Enter email address")
                last_name = st.text_input("Last Name *", placeholder="Enter last name")
                send_welcome = st.checkbox("Send welcome email", value=True)
            
            password = st.text_input("Password *", type="password", placeholder="Enter password")
            confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Confirm password")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit_btn = st.form_submit_button("Create User", type="primary")
            
            with col2:
                cancel_btn = st.form_submit_button("Cancel")
            
            if cancel_btn:
                st.session_state.show_add_user_modal = False
                st.rerun()
            
            if submit_btn:
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif not all([username, email, first_name, last_name, password]):
                    st.error("Please fill in all required fields!")
                else:
                    success = self._create_user(username, email, password, first_name, last_name, role)
                    if success:
                        st.success(f"User '{username}' created successfully!")
                        st.session_state.show_add_user_modal = False
                        time.sleep(2)
                        st.rerun()
    
    def _get_users_data(self, search_term: str, role_filter: str, status_filter: str) -> pd.DataFrame:
        """Get users data with filters."""
        # Sample user data
        users = []
        for i in range(1, 51):
            users.append({
                'id': f'user_{i:03d}',
                'username': f'user_{i:02d}',
                'email': f'user_{i:02d}@example.com',
                'full_name': f'User {i:02d} Name',
                'role': np.random.choice(['admin', 'manager', 'user', 'guest'], p=[0.1, 0.2, 0.6, 0.1]),
                'status': np.random.choice(['Active', 'Inactive', 'Locked'], p=[0.8, 0.15, 0.05]),
                'last_login': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
                'created_at': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d')
            })
        
        df = pd.DataFrame(users)
        
        # Apply search filter
        if search_term:
            mask = (
                df['username'].str.contains(search_term, case=False, na=False) |
                df['email'].str.contains(search_term, case=False, na=False) |
                df['full_name'].str.contains(search_term, case=False, na=False)
            )
            df = df[mask]
        
        # Apply role filter
        if role_filter != "All":
            df = df[df['role'] == role_filter.lower()]
        
        # Apply status filter
        if status_filter != "All":
            df = df[df['status'] == status_filter]
        
        return df
    
    def _create_user(self, username: str, email: str, password: str, 
                    first_name: str, last_name: str, role: str) -> bool:
        """Create a new user."""
        try:
            # In a real implementation, this would call the UserService
            # For demo purposes, we'll simulate success
            return True
        except Exception as e:
            st.error(f"Error creating user: {str(e)}")
            return False
    
    def _edit_user(self, user_id: str):
        """Edit user details."""
        st.session_state[f'editing_user_{user_id}'] = True
        st.info(f"Edit user functionality for {user_id} - Coming soon!")
    
    def _toggle_user_lock(self, user_id: str):
        """Toggle user lock status."""
        st.info(f"Toggle lock for user {user_id} - Coming soon!")
    
    def _delete_user(self, user_id: str):
        """Delete a user."""
        st.warning(f"Delete user {user_id} - Coming soon!")
    
    def _export_users(self):
        """Export users data."""
        users_data = self._get_users_data("", "All", "All")
        csv = users_data.to_csv(index=False)
        
        st.download_button(
            label="Download Users CSV",
            data=csv,
            file_name=f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _render_roles_section(self):
        """Render roles management section."""
        st.subheader("User Roles")
        
        # Role definitions
        roles_data = [
            {
                'role': 'Super Admin',
                'description': 'Full system access and administration',
                'permissions': ['*'],
                'users_count': 2,
                'color': '#dc3545'
            },
            {
                'role': 'Admin',
                'description': 'Administrative access with some restrictions',
                'permissions': ['user_management', 'system_settings', 'reports'],
                'users_count': 5,
                'color': '#fd7e14'
            },
            {
                'role': 'Manager',
                'description': 'Management access to assigned areas',
                'permissions': ['team_management', 'reports', 'data_export'],
                'users_count': 12,
                'color': '#ffc107'
            },
            {
                'role': 'User',
                'description': 'Standard user access',
                'permissions': ['read_data', 'create_reports'],
                'users_count': 28,
                'color': '#28a745'
            },
            {
                'role': 'Guest',
                'description': 'Limited read-only access',
                'permissions': ['read_public_data'],
                'users_count': 3,
                'color': '#6c757d'
            }
        ]
        
        for role_info in roles_data:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 3, 1, 2])
                
                with col1:
                    st.markdown(f"**<span style='color: {role_info['color']}'>{role_info['role']}</span>**", 
                              unsafe_allow_html=True)
                    st.caption(f"{role_info['users_count']} users")
                
                with col2:
                    st.write(role_info['description'])
                    permissions_str = ", ".join(role_info['permissions'][:3])
                    if len(role_info['permissions']) > 3:
                        permissions_str += f" +{len(role_info['permissions']) - 3} more"
                    st.caption(f"Permissions: {permissions_str}")
                
                with col3:
                    # Role usage chart
                    fig = go.Figure(data=[go.Pie(
                        values=[role_info['users_count'], 50 - role_info['users_count']],
                        labels=['Assigned', 'Available'],
                        hole=0.7,
                        showlegend=False,
                        marker_colors=[role_info['color'], '#f8f9fa']
                    )])
                    
                    fig.update_layout(
                        height=80,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[dict(text=str(role_info['users_count']), 
                                        x=0.5, y=0.5, font_size=12, showarrow=False)]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("✏️ Edit", key=f"edit_role_{role_info['role']}"):
                            st.info(f"Edit role {role_info['role']} - Coming soon!")
                    
                    with action_col2:
                        if st.button("👥 Users", key=f"view_users_{role_info['role']}"):
                            st.info(f"View users with role {role_info['role']} - Coming soon!")
                
                st.divider()
    
    def _render_permissions_section(self):
        """Render permissions management section."""
        st.subheader("System Permissions")
        
        # Permission categories
        permission_categories = {
            'User Management': [
                'create_user', 'read_user', 'update_user', 'delete_user',
                'manage_roles', 'assign_permissions'
            ],
            'Data Management': [
                'import_data', 'export_data', 'delete_data', 'backup_data',
                'restore_data'
            ],
            'System Administration': [
                'system_settings', 'view_logs', 'system_health',
                'manage_notifications', 'system_maintenance'
            ],
            'Business Operations': [
                'create_reports', 'view_analytics', 'manage_workflows',
                'process_transactions'
            ]
        }
        
        for category, permissions in permission_categories.items():
            with st.expander(f"📋 {category} ({len(permissions)} permissions)"):
                
                for i in range(0, len(permissions), 3):
                    cols = st.columns(3)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(permissions):
                            perm = permissions[i + j]
                            with col:
                                # Permission card
                                st.markdown(f"""
                                <div style="
                                    border: 1px solid #e0e0e0;
                                    border-radius: 5px;
                                    padding: 10px;
                                    margin: 5px 0;
                                    background: white;
                                ">
                                    <strong>{perm.replace('_', ' ').title()}</strong><br>
                                    <small style="color: #666;">Permission: {perm}</small>
                                </div>
                                """, unsafe_allow_html=True)
        
        # Permission assignment matrix
        st.subheader("Permission Matrix")
        
        roles = ['Super Admin', 'Admin', 'Manager', 'User', 'Guest']
        all_perms = []
        for perms in permission_categories.values():
            all_perms.extend(perms)
        
        # Create sample permission matrix
        matrix_data = []
        for perm in all_perms[:10]:  # Show first 10 permissions
            row = {'Permission': perm.replace('_', ' ').title()}
            for role in roles:
                # Simulate permission assignments
                if role == 'Super Admin':
                    row[role] = '✅'
                elif role == 'Admin':
                    row[role] = '✅' if 'user' in perm or 'system' in perm else '❌'
                elif role == 'Manager':
                    row[role] = '✅' if 'create' in perm or 'read' in perm else '❌'
                elif role == 'User':
                    row[role] = '✅' if 'read' in perm or 'create_reports' in perm else '❌'
                else:  # Guest
                    row[role] = '✅' if 'read' in perm else '❌'
            
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    
    def _render_user_analytics(self):
        """Render user analytics section."""
        st.subheader("User Analytics")
        
        # Generate sample analytics data
        col1, col2 = st.columns(2)
        
        with col1:
            # User registrations over time
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
            registrations = np.cumsum(np.random.poisson(3, len(dates)))
            
            fig = px.line(
                x=dates,
                y=registrations,
                title='User Registrations Over Time',
                labels={'x': 'Date', 'y': 'Total Users'}
            )
            fig.update_traces(line_color='#1f77b4')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Role distribution pie chart
            role_counts = [2, 5, 12, 28, 3]  # Super Admin, Admin, Manager, User, Guest
            role_names = ['Super Admin', 'Admin', 'Manager', 'User', 'Guest']
            
            fig = px.pie(
                values=role_counts,
                names=role_names,
                title='User Role Distribution',
                color_discrete_sequence=['#dc3545', '#fd7e14', '#ffc107', '#28a745', '#6c757d']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity heatmap
        st.subheader("User Activity Heatmap")
        
        # Generate sample activity data
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        activity_matrix = np.random.poisson(5, (7, 24))
        
        fig = go.Figure(data=go.Heatmap(
            z=activity_matrix,
            x=hours,
            y=days,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Active Users")
        ))
        
        fig.update_layout(
            title="User Activity by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # User statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Statistics")
            
            stats = [
                ("Total Users", "50"),
                ("Active Users (30 days)", "42"),
                ("New Users (7 days)", "3"),
                ("Average Session Duration", "24 min"),
                ("Most Active Role", "User"),
            ]
            
            for stat, value in stats:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.write(stat)
                with col_b:
                    st.metric("", value)
        
        with col2:
            st.subheader("Top Active Users")
            
            active_users = [
                {"User": "admin", "Sessions": 156, "Last Seen": "2 min ago"},
                {"User": "john_doe", "Sessions": 89, "Last Seen": "1 hour ago"},
                {"User": "jane_smith", "Sessions": 67, "Last Seen": "3 hours ago"},
                {"User": "manager_01", "Sessions": 45, "Last Seen": "1 day ago"},
                {"User": "analyst_02", "Sessions": 34, "Last Seen": "2 days ago"},
            ]
            
            active_df = pd.DataFrame(active_users)
            st.dataframe(active_df, use_container_width=True, hide_index=True)


# ==================== REPORTS AND ANALYTICS ====================

class ReportsPage:
    """Reports and analytics page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render reports page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>📈 Reports & Analytics</h1>
            <p>Business Intelligence and Data Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Business Reports", "📈 Analytics", "🔍 Custom Reports", "📅 Scheduled Reports"])
        
        with tab1:
            self._render_business_reports()
        
        with tab2:
            self._render_analytics_dashboard()
        
        with tab3:
            self._render_custom_reports()
        
        with tab4:
            self._render_scheduled_reports()
    
    def _render_business_reports(self):
        """Render business reports section."""
        st.subheader("Standard Business Reports")
        
        # Report categories
        report_categories = {
            "Financial Reports": [
                {"name": "Profit & Loss Statement", "description": "Monthly P&L analysis", "icon": "💰"},
                {"name": "Balance Sheet", "description": "Assets, liabilities and equity", "icon": "⚖️"},
                {"name": "Cash Flow Statement", "description": "Cash inflows and outflows", "icon": "💵"},
                {"name": "Revenue Analysis", "description": "Revenue trends and forecasting", "icon": "📈"},
            ],
            "Operational Reports": [
                {"name": "User Activity Report", "description": "System usage and engagement", "icon": "👥"},
                {"name": "Performance Metrics", "description": "System performance indicators", "icon": "⚡"},
                {"name": "Data Quality Report", "description": "Data integrity and completeness", "icon": "🎯"},
                {"name": "Security Audit", "description": "Security events and compliance", "icon": "🔒"},
            ],
            "Management Reports": [
                {"name": "Executive Dashboard", "description": "High-level KPIs and metrics", "icon": "🎛️"},
                {"name": "Department Summary", "description": "Department-wise performance", "icon": "🏢"},
                {"name": "Resource Utilization", "description": "Resource allocation and usage", "icon": "📊"},
                {"name": "Compliance Report", "description": "Regulatory compliance status", "icon": "📋"},
            ]
        }
        
        for category, reports in report_categories.items():
            st.markdown(f"### {category}")
            
            cols = st.columns(2)
            for i, report in enumerate(reports):
                with cols[i % 2]:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            st.markdown(f"<h2 style='text-align: center;'>{report['icon']}</h2>", 
                                      unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**{report['name']}**")
                            st.caption(report['description'])
                        
                        with col3:
                            if st.button("Generate", key=f"gen_{report['name']}"):
                                self._generate_report(report['name'])
                        
                        st.divider()
    
    def _generate_report(self, report_name: str):
        """Generate a specific report."""
        with st.spinner(f"Generating {report_name}..."):
            time.sleep(2)  # Simulate report generation
            
            if "Profit & Loss" in report_name:
                self._show_pl_report()
            elif "User Activity" in report_name:
                self._show_user_activity_report()
            elif "Performance Metrics" in report_name:
                self._show_performance_report()
            else:
                st.success(f"{report_name} generated successfully!")
                st.info("Report content would be displayed here.")
    
    def _show_pl_report(self):
        """Show Profit & Loss report."""
        st.subheader("Profit & Loss Statement")
        
        # Generate sample P&L data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        pl_data = {
            'Month': months,
            'Revenue': np.random.uniform(50000, 150000, 12),
            'Cost of Goods Sold': np.random.uniform(20000, 60000, 12),
            'Operating Expenses': np.random.uniform(15000, 40000, 12),
            'Interest Expense': np.random.uniform(1000, 5000, 12),
            'Tax Expense': np.random.uniform(2000, 8000, 12)
        }
        
        df = pd.DataFrame(pl_data)
        df['Gross Profit'] = df['Revenue'] - df['Cost of Goods Sold']
        df['Operating Income'] = df['Gross Profit'] - df['Operating Expenses']
        df['Net Income'] = df['Operating Income'] - df['Interest Expense'] - df['Tax Expense']
        
        # Display P&L table
        st.dataframe(df.round(2), use_container_width=True, hide_index=True)
        
        # P&L visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Month'], y=df['Revenue'],
            mode='lines+markers', name='Revenue',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Month'], y=df['Net Income'],
            mode='lines+markers', name='Net Income',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Revenue vs Net Income Trend',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_user_activity_report(self):
        """Show user activity report."""
        st.subheader("User Activity Report")
        
        # Generate sample activity data
        dates = pd.date_range(start='2024-11-01', end='2024-12-31', freq='D')
        
        activity_data = {
            'Date': dates,
            'Daily Active Users': np.random.poisson(25, len(dates)),
            'New Registrations': np.random.poisson(2, len(dates)),
            'Sessions': np.random.poisson(150, len(dates)),
            'Avg Session Duration (min)': np.random.normal(20, 5, len(dates))
        }
        
        df = pd.DataFrame(activity_data)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Active Users", f"{df['Daily Active Users'].mean():.0f}", 
                     delta=f"+{np.random.randint(1, 5)}")
        
        with col2:
            st.metric("Total Sessions", f"{df['Sessions'].sum():,}", 
                     delta=f"+{np.random.randint(50, 200)}")
        
        with col3:
            st.metric("Avg Session Duration", f"{df['Avg Session Duration (min)'].mean():.1f} min",
                     delta=f"+{np.random.uniform(0.5, 2.0):.1f}")
        
        with col4:
            st.metric("New Users", f"{df['New Registrations'].sum()}", 
                     delta=f"+{np.random.randint(5, 15)}")
        
        # Activity trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Active Users', 'Session Count', 'New Registrations', 'Session Duration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Daily Active Users'], name='DAU'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Sessions'], name='Sessions'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['New Registrations'], name='New Users'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Avg Session Duration (min)'], name='Duration'), row=2, col=2)
        
        fig.update_layout(height=500, title_text="User Activity Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_performance_report(self):
        """Show system performance report."""
        st.subheader("System Performance Report")
        
        # Generate sample performance data
        timestamps = pd.date_range(start='2024-12-01', periods=100, freq='H')
        
        perf_data = {
            'Timestamp': timestamps,
            'CPU Usage (%)': np.random.uniform(20, 80, 100),
            'Memory Usage (%)': np.random.uniform(30, 90, 100),
            'Response Time (ms)': np.random.exponential(100, 100),
            'Requests/sec': np.random.poisson(50, 100),
            'Error Rate (%)': np.random.exponential(0.5, 100)
        }
        
        df = pd.DataFrame(perf_data)
        
        # Performance gauges
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_cpu = df['CPU Usage (%)'].mean()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_cpu,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_memory = df['Memory Usage (%)'].mean()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_memory,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "green"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            avg_response = df['Response Time (ms)'].mean()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_response,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Response Time (ms)"},
                gauge={'axis': {'range': [0, 500]},
                       'bar': {'color': "orange"},
                       'steps': [{'range': [0, 100], 'color': "lightgray"},
                                {'range': [100, 300], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 400}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU & Memory Usage', 'Response Time', 'Request Rate', 'Error Rate')
        )
        
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['CPU Usage (%)'], name='CPU %'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Memory Usage (%)'], name='Memory %'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Response Time (ms)'], name='Response Time'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Requests/sec'], name='Requests/sec'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Error Rate (%)'], name='Error Rate %'), row=2, col=2)
        
        fig.update_layout(height=500, title_text="Performance Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_analytics_dashboard(self):
        """Render analytics dashboard."""
        st.subheader("Advanced Analytics Dashboard")
        
        # Analytics filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=[date.today() - timedelta(days=30), date.today()],
                key="analytics_date_range"
            )
        
        with col2:
            metric_type = st.selectbox(
                "Metric Type",
                ["Business", "Technical", "User", "Financial"],
                key="analytics_metric_type"
            )
        
        with col3:
            granularity = st.selectbox(
                "Granularity",
                ["Daily", "Weekly", "Monthly"],
                key="analytics_granularity"
            )
        
        # Generate analytics based on selection
        if metric_type == "Business":
            self._render_business_analytics()
        elif metric_type == "Technical":
            self._render_technical_analytics()
        elif metric_type == "User":
            self._render_user_analytics_detailed()
        else:
            self._render_financial_analytics()
    
    def _render_business_analytics(self):
        """Render business analytics."""
        st.markdown("#### Business Analytics")
        
        # Key business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Revenue Growth", "12.5%", "↗️"),
            ("Customer Acquisition", "156", "👥"),
            ("Conversion Rate", "3.2%", "🎯"),
            ("Churn Rate", "2.1%", "📉")
        ]
        
        for i, (label, value, icon) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.metric(
                    f"{icon} {label}",
                    value,
                    delta=f"+{np.random.uniform(0.1, 2.0):.1f}%" if "Growth" in label else None
                )
        
        # Business trends
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        business_data = pd.DataFrame({
            'Month': dates,
            'Revenue': np.cumsum(np.random.normal(10000, 2000, 12)) + 50000,
            'Customers': np.cumsum(np.random.poisson(20, 12)) + 100,
            'Orders': np.random.poisson(500, 12) + 300
        })
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Customer Growth', 'Order Volume', 'Revenue vs Customers'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        fig.add_trace(go.Scatter(x=business_data['Month'], y=business_data['Revenue'], name='Revenue'), row=1, col=1)
        fig.add_trace(go.Scatter(x=business_data['Month'], y=business_data['Customers'], name='Customers'), row=1, col=2)
        fig.add_trace(go.Bar(x=business_data['Month'], y=business_data['Orders'], name='Orders'), row=2, col=1)
        
        # Correlation plot
        fig.add_trace(go.Scatter(x=business_data['Revenue'], y=business_data['Customers'], 
                               mode='markers', name='Revenue vs Customers'), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Business Performance Analytics")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_technical_analytics(self):
        """Render technical analytics."""
        st.markdown("#### Technical Analytics")
        
        # System health indicators
        health_metrics = [
            ("System Uptime", "99.8%", "🟢"),
            ("API Response Time", "145ms", "⚡"),
            ("Error Rate", "0.02%", "🔴"),
            ("Database Connections", "47/100", "🗄️")
        ]
        
        cols = st.columns(4)
        for i, (label, value, icon) in enumerate(health_metrics):
            with cols[i]:
                st.metric(f"{icon} {label}", value)
        
        # Technical performance heatmap
        st.subheader("System Performance Heatmap")
        
        components = ['API Server', 'Database', 'Cache', 'File Storage', 'Message Queue']
        time_periods = [f"{i:02d}:00" for i in range(24)]
        
        # Generate sample performance data (0-100 scale)
        performance_matrix = np.random.uniform(75, 100, (len(components), len(time_periods)))
        
        # Add some performance dips
        performance_matrix[1, 8:10] = np.random.uniform(60, 75, 2)  # Database morning dip
        performance_matrix[0, 14:16] = np.random.uniform(65, 80, 2)  # API afternoon load
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=time_periods,
            y=components,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Performance Score")
        ))
        
        fig.update_layout(
            title="24-Hour System Performance",
            xaxis_title="Hour of Day",
            yaxis_title="System Components",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_user_analytics_detailed(self):
        """Render detailed user analytics."""
        st.markdown("#### User Behavior Analytics")
        
        # User engagement funnel
        funnel_data = {
            'Stage': ['Visitors', 'Sign-ups', 'Activated', 'Engaged', 'Retained'],
            'Count': [10000, 2500, 2000, 1200, 800],
            'Conversion': [100, 25, 20, 12, 8]
        }
        
        funnel_df = pd.DataFrame(funnel_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Funnel(
                y=funnel_df['Stage'],
                x=funnel_df['Count'],
                textinfo="value+percent initial"
            ))
            fig.update_layout(title="User Engagement Funnel", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # User cohort analysis
            cohort_data = np.random.uniform(0.6, 1.0, (6, 12))  # 6 months x 12 weeks
            cohort_data = np.triu(cohort_data)  # Upper triangular matrix
            
            fig = go.Figure(data=go.Heatmap(
                z=cohort_data,
                x=[f"Week {i+1}" for i in range(12)],
                y=[f"Cohort {i+1}" for i in range(6)],
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title="User Retention Cohorts",
                xaxis_title="Period",
                yaxis_title="Cohort",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_financial_analytics(self):
        """Render financial analytics."""
        st.markdown("#### Financial Analytics")
        
        # Financial KPIs
        kpi_data = [
            ("Monthly Recurring Revenue", "$125,430", "💰"),
            ("Customer Lifetime Value", "$2,340", "👤"),
            ("Cost Per Acquisition", "$89", "📊"),
            ("Gross Margin", "68.5%", "📈")
        ]
        
        cols = st.columns(4)
        for i, (label, value, icon) in enumerate(kpi_data):
            with cols[i]:
                delta_val = f"+{np.random.uniform(1, 15):.1f}%"
                st.metric(f"{icon} {label}", value, delta=delta_val)
        
        # Financial projections
        months = pd.date_range(start='2024-01-01', periods=18, freq='M')
        
        # Historical data (first 12 months)
        historical_revenue = np.cumsum(np.random.normal(8000, 1500, 12)) + 50000
        historical_expenses = historical_revenue * np.random.uniform(0.6, 0.8, 12)
        
        # Projected data (next 6 months)
        projected_revenue = historical_revenue[-1] + np.cumsum(np.random.normal(10000, 2000, 6))
        projected_expenses = projected_revenue * np.random.uniform(0.65, 0.75, 6)
        
        # Combine historical and projected
        all_revenue = np.concatenate([historical_revenue, projected_revenue])
        all_expenses = np.concatenate([historical_expenses, projected_expenses])
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=months[:12], y=historical_revenue,
            mode='lines+markers', name='Historical Revenue',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months[:12], y=historical_expenses,
            mode='lines+markers', name='Historical Expenses',
            line=dict(color='red', width=3)
        ))
        
        # Projected data
        fig.add_trace(go.Scatter(
            x=months[12:], y=projected_revenue,
            mode='lines+markers', name='Projected Revenue',
            line=dict(color='lightblue', width=3, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=months[12:], y=projected_expenses,
            mode='lines+markers', name='Projected Expenses',
            line=dict(color='lightcoral', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Financial Projections (Historical vs Projected)',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_custom_reports(self):
        """Render custom reports builder."""
        st.subheader("Custom Report Builder")
        
        with st.form("custom_report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_name = st.text_input("Report Name", placeholder="Enter report name...")
                
                data_source = st.selectbox(
                    "Data Source",
                    ["Users", "Transactions", "System Logs", "Performance Metrics", "Custom Query"]
                )
                
                chart_type = st.selectbox(
                    "Visualization Type",
                    ["Table", "Line Chart", "Bar Chart", "Pie Chart", "Heatmap", "Scatter Plot"]
                )
            
            with col2:
                date_column = st.selectbox(
                    "Date Column",
                    ["created_at", "updated_at", "processed_at", "logged_at"]
                )
                
                group_by = st.multiselect(
                    "Group By",
                    ["date", "user_type", "category", "status", "department"]
                )
                
                aggregation = st.selectbox(
                    "Aggregation",
                    ["Count", "Sum", "Average", "Min", "Max", "Median"]
                )
            
            # Filters section
            st.subheader("Filters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_column = st.selectbox("Filter Column", ["status", "type", "category", "user_id"])
            
            with col2:
                filter_operator = st.selectbox("Operator", ["equals", "not_equals", "contains", "in"])
            
            with col3:
                filter_value = st.text_input("Filter Value")
            
            # Advanced options
            with st.expander("Advanced Options"):
                include_totals = st.checkbox("Include Totals")
                show_percentages = st.checkbox("Show Percentages")
                export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF"])
            
            # Generate report button
            col1, col2 = st.columns([1, 3])
            
            with col1:
                generate_btn = st.form_submit_button("Generate Report", type="primary")
            
            if generate_btn and report_name:
                with st.spinner("Generating custom report..."):
                    time.sleep(2)
                    
                    # Generate sample data based on selections
                    sample_data = self._generate_sample_report_data(data_source, group_by)
                    
                    st.success(f"Custom report '{report_name}' generated successfully!")
                    
                    # Display generated report
                    st.subheader(f"📊 {report_name}")
                    
                    if chart_type == "Table":
                        st.dataframe(sample_data, use_container_width=True)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(sample_data, x=sample_data.columns[0], y=sample_data.columns[1])
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Line Chart":
                        fig = px.line(sample_data, x=sample_data.columns[0], y=sample_data.columns[1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    if export_format == "CSV":
                        csv = sample_data.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            file_name=f"{report_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
    
    def _generate_sample_report_data(self, data_source: str, group_by: List[str]) -> pd.DataFrame:
        """Generate sample data for custom reports."""
        if data_source == "Users":
            return pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
                'User Count': np.random.poisson(25, 30),
                'Category': np.random.choice(['New', 'Returning', 'Premium'], 30)
            })
        elif data_source == "Transactions":
            return pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
                'Amount': np.random.uniform(1000, 50000, 30),
                'Status': np.random.choice(['Completed', 'Pending', 'Failed'], 30)
            })
        else:
            return pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
                'Value': np.random.uniform(50, 200, 30),
                'Type': np.random.choice(['Type A', 'Type B', 'Type C'], 30)
            })
    
    def _render_scheduled_reports(self):
        """Render scheduled reports section."""
        st.subheader("Scheduled Reports")
        
        # Add new scheduled report
        with st.expander("➕ Schedule New Report"):
            with st.form("schedule_report_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    report_name = st.text_input("Report Name")
                    report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Analytics", "Monthly Report"])
                    recipients = st.text_area("Email Recipients (one per line)")
                
                with col2:
                    schedule_type = st.selectbox("Schedule", ["Daily", "Weekly", "Monthly"])
                    time_to_send = st.time_input("Send Time", value=datetime.now().time())
                    format_type = st.selectbox("Format", ["PDF", "Excel", "CSV"])
                
                if st.form_submit_button("Schedule Report"):
                    st.success(f"Report '{report_name}' scheduled successfully!")
        
        # Existing scheduled reports
        st.subheader("Current Scheduled Reports")
        
        scheduled_reports = [
            {
                "Report": "Daily User Activity",
                "Schedule": "Daily at 08:00",
                "Recipients": "admin@company.com, manager@company.com",
                "Format": "PDF",
                "Status": "Active",
                "Next Run": "Tomorrow 08:00"
            },
            {
                "Report": "Weekly Financial Summary",
                "Schedule": "Weekly on Monday",
                "Recipients": "finance@company.com",
                "Format": "Excel",
                "Status": "Active",
                "Next Run": "Monday 09:00"
            },
            {
                "Report": "Monthly Performance Report",
                "Schedule": "Monthly 1st day",
                "Recipients": "executives@company.com",
                "Format": "PDF",
                "Status": "Paused",
                "Next Run": "Next month"
            }
        ]
        
        for i, report in enumerate(scheduled_reports):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
                
                with col1:
                    st.write(f"**{report['Report']}**")
                    st.caption(f"Format: {report['Format']} | {report['Schedule']}")
                
                with col2:
                    st.write(f"Next: {report['Next Run']}")
                    st.caption(f"To: {report['Recipients'][:30]}...")
                
                with col3:
                    status_color = "🟢" if report['Status'] == 'Active' else "⏸️"
                    st.write(f"{status_color} {report['Status']}")
                
                with col4:
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("✏️", key=f"edit_sched_{i}",help="Edit schedule"):
                            st.info(f"Edit schedule for {report['Report']} - Coming soon!")
                    
                    with action_col2:
                        pause_resume = "▶️" if report['Status'] == 'Paused' else "⏸️"
                        if st.button(pause_resume, key=f"toggle_sched_{i}", 
                                   help="Pause/Resume"):
                            st.info(f"Toggle schedule for {report['Report']} - Coming soon!")
                    
                    with action_col3:
                        if st.button("🗑️", key=f"delete_sched_{i}", help="Delete"):
                            st.warning(f"Delete schedule for {report['Report']} - Coming soon!")
                
                st.divider()


# ==================== NOTIFICATIONS ====================

class NotificationsPage:
    """Notifications page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render notifications page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>🔔 Notifications</h1>
            <p>System notifications and alerts management</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📨 Inbox", "✉️ Send Notification", "📋 Templates", "⚙️ Settings"])
        
        with tab1:
            self._render_notifications_inbox()
        
        with tab2:
            self._render_send_notification()
        
        with tab3:
            self._render_notification_templates()
        
        with tab4:
            self._render_notification_settings()
    
    def _render_notifications_inbox(self):
        """Render notifications inbox."""
        # Inbox controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.subheader("Notification Inbox")
        
        with col2:
            if st.button("🔄 Refresh", key="refresh_notifications"):
                st.rerun()
        
        with col3:
            if st.button("📧 Mark All Read", key="mark_all_read"):
                st.success("All notifications marked as read!")
        
        with col4:
            if st.button("🗑️ Clear All", key="clear_all_notifications"):
                st.warning("Clear all functionality - Coming soon!")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox("Status", ["All", "Unread", "Read"], key="notif_status_filter")
        
        with col2:
            priority_filter = st.selectbox("Priority", ["All", "High", "Medium", "Low"], key="notif_priority_filter")
        
        with col3:
            type_filter = st.selectbox("Type", ["All", "System", "User", "Security", "Business"], key="notif_type_filter")
        
        # Generate sample notifications
        notifications = self._generate_sample_notifications()
        
        # Apply filters
        filtered_notifications = self._apply_notification_filters(
            notifications, status_filter, priority_filter, type_filter
        )
        
        # Display notifications
        if filtered_notifications:
            st.markdown("### Recent Notifications")
            
            for notif in filtered_notifications:
                self._render_notification_item(notif)
        
        else:
            st.info("No notifications match your filter criteria.")
    
    def _generate_sample_notifications(self) -> List[Dict[str, Any]]:
        """Generate sample notifications."""
        notifications = [
            {
                "id": "1",
                "title": "System Backup Completed",
                "message": "Daily system backup completed successfully at 02:00 AM",
                "type": "System",
                "priority": "Low",
                "status": "Unread",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "icon": "💾",
                "action_url": "/system/backup"
            },
            {
                "id": "2", 
                "title": "New User Registration",
                "message": "User 'john.doe@company.com' has registered and pending approval",
                "type": "User",
                "priority": "Medium",
                "status": "Unread",
                "timestamp": datetime.now() - timedelta(hours=1),
                "icon": "👤",
                "action_url": "/users/pending"
            },
            {
                "id": "3",
                "title": "Security Alert: Multiple Failed Logins",
                "message": "5 failed login attempts detected from IP 192.168.1.100",
                "type": "Security",
                "priority": "High",
                "status": "Read",
                "timestamp": datetime.now() - timedelta(hours=2),
                "icon": "🔒",
                "action_url": "/security/alerts"
            },
            {
                "id": "4",
                "title": "Monthly Report Generated",
                "message": "Financial report for November 2024 is ready for download",
                "type": "Business",
                "priority": "Medium",
                "status": "Read",
                "timestamp": datetime.now() - timedelta(hours=4),
                "icon": "📊",
                "action_url": "/reports/financial"
            },
            {
                "id": "5",
                "title": "System Maintenance Scheduled",
                "message": "Scheduled maintenance on Sunday 2024-12-15 from 02:00 to 04:00 AM",
                "type": "System",
                "priority": "High",
                "status": "Unread",
                "timestamp": datetime.now() - timedelta(hours=6),
                "icon": "⚠️",
                "action_url": "/system/maintenance"
            }
        ]
        
        return notifications
    
    def _apply_notification_filters(self, notifications: List[Dict], status: str, priority: str, type_filter: str) -> List[Dict]:
        """Apply filters to notifications."""
        filtered = notifications
        
        if status != "All":
            filtered = [n for n in filtered if n["status"] == status]
        
        if priority != "All":
            filtered = [n for n in filtered if n["priority"] == priority]
        
        if type_filter != "All":
            filtered = [n for n in filtered if n["type"] == type_filter]
        
        return filtered
    
    def _render_notification_item(self, notification: Dict[str, Any]):
        """Render a single notification item."""
        # Determine styling based on status and priority
        bg_color = "#f8f9fa" if notification["status"] == "Read" else "#fff3cd"
        border_color = {
            "High": "#dc3545",
            "Medium": "#ffc107", 
            "Low": "#28a745"
        }.get(notification["priority"], "#6c757d")
        
        # Time formatting
        time_ago = self._format_time_ago(notification["timestamp"])
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 4px solid {border_color};
                background-color: {bg_color};
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex-grow: 1;">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.2em; margin-right: 10px;">{notification["icon"]}</span>
                            <strong style="font-size: 1.1em;">{notification["title"]}</strong>
                            <span style="
                                background-color: {border_color};
                                color: white;
                                padding: 2px 8px;
                                border-radius: 12px;
                                font-size: 0.8em;
                                margin-left: 10px;
                            ">{notification["priority"]}</span>
                        </div>
                        <p style="margin: 8px 0; color: #333;">{notification["message"]}</p>
                        <small style="color: #666;">
                            {notification["type"]} • {time_ago} • {notification["status"]}
                        </small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            
            with col1:
                if st.button("👀 View", key=f"view_{notification['id']}"):
                    st.info(f"View details for notification {notification['id']} - Coming soon!")
            
            with col2:
                mark_text = "📧 Mark Read" if notification["status"] == "Unread" else "📭 Mark Unread"
                if st.button(mark_text, key=f"mark_{notification['id']}"):
                    st.success(f"Notification marked as {'read' if notification['status'] == 'Unread' else 'unread'}!")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{notification['id']}"):
                    st.warning(f"Delete notification {notification['id']} - Coming soon!")
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    
    def _render_send_notification(self):
        """Render send notification form."""
        st.subheader("Send New Notification")
        
        with st.form("send_notification_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                notification_type = st.selectbox(
                    "Notification Type",
                    ["System Alert", "User Message", "Security Notice", "Business Update"]
                )
                
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                
                recipients = st.text_area(
                    "Recipients",
                    placeholder="Enter email addresses, one per line or comma-separated"
                )
            
            with col2:
                delivery_method = st.multiselect(
                    "Delivery Method",
                    ["Email", "In-App", "SMS", "Push Notification"],
                    default=["Email", "In-App"]
                )
                
                schedule_option = st.selectbox(
                    "Send Option",
                    ["Send Now", "Schedule for Later"]
                )
                
                if schedule_option == "Schedule for Later":
                    scheduled_time = st.datetime_input(
                        "Schedule Time",
                        value=datetime.now() + timedelta(hours=1)
                    )
            
            # Message content
            st.subheader("Message Content")
            
            subject = st.text_input("Subject", placeholder="Enter notification subject")
            
            message = st.text_area(
                "Message", 
                placeholder="Enter notification message...",
                height=150
            )
            
            # Additional options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    include_action_button = st.checkbox("Include Action Button")
                    if include_action_button:
                        action_text = st.text_input("Button Text", value="View Details")
                        action_url = st.text_input("Button URL")
                
                with col2:
                    auto_expire = st.checkbox("Auto Expire")
                    if auto_expire:
                        expire_days = st.number_input("Expire After (days)", value=7, min_value=1)
                    
                    track_delivery = st.checkbox("Track Delivery Status", value=True)
            
            # Send button
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                send_btn = st.form_submit_button("📤 Send Notification", type="primary")
            
            with col2:
                preview_btn = st.form_submit_button("👁️ Preview")
            
            if send_btn and subject and message and recipients:
                with st.spinner("Sending notification..."):
                    time.sleep(2)  # Simulate sending
                    
                    recipient_count = len([r.strip() for r in recipients.replace(',', '\n').split('\n') if r.strip()])
                    
                    st.success(f"Notification '{subject}' sent to {recipient_count} recipients via {', '.join(delivery_method)}!")
                    
                    # Show sending summary
                    st.json({
                        "Subject": subject,
                        "Recipients": recipient_count,
                        "Delivery Methods": delivery_method,
                        "Priority": priority,
                        "Status": "Sent" if schedule_option == "Send Now" else "Scheduled"
                    })
            
            elif preview_btn and subject and message:
                st.subheader("📋 Notification Preview")
                
                # Preview notification
                st.markdown(f"""
                <div style="
                    border: 2px solid #007bff;
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #f8f9fa;
                    margin: 10px 0;
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <span style="font-size: 1.5em; margin-right: 10px;">🔔</span>
                        <strong style="font-size: 1.2em;">{subject}</strong>
                        <span style="
                            background-color: {'#dc3545' if priority == 'Critical' else '#ffc107' if priority == 'High' else '#28a745'};
                            color: white;
                            padding: 3px 10px;
                            border-radius: 15px;
                            font-size: 0.8em;
                            margin-left: 15px;
                        ">{priority}</span>
                    </div>
                    <p style="margin: 10px 0; line-height: 1.5;">{message}</p>
                    <hr>
                    <small style="color: #666;">
                        Type: {notification_type} | Delivery: {', '.join(delivery_method)} | 
                        Recipients: {len([r.strip() for r in recipients.replace(',', '\n').split('\n') if r.strip()]) if recipients else 0}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            elif send_btn:
                st.error("Please fill in all required fields (Subject, Message, Recipients)")
    
    def _render_notification_templates(self):
        """Render notification templates."""
        st.subheader("Notification Templates")
        
        # Add new template
        with st.expander("➕ Create New Template"):
            with st.form("create_template_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    template_name = st.text_input("Template Name")
                    template_category = st.selectbox(
                        "Category",
                        ["System", "Security", "Business", "User Management", "Reports"]
                    )
                
                with col2:
                    default_priority = st.selectbox("Default Priority", ["Low", "Medium", "High"])
                    default_channels = st.multiselect(
                        "Default Channels",
                        ["Email", "In-App", "SMS", "Push"],
                        default=["Email", "In-App"]
                    )
                
                template_subject = st.text_input("Subject Template", placeholder="Use {variable_name} for variables")
                template_body = st.text_area("Body Template", height=100, placeholder="Use {variable_name} for variables")
                
                variables = st.text_input("Variables (comma-separated)", placeholder="user_name, action_type, timestamp")
                
                if st.form_submit_button("Create Template"):
                    if template_name and template_subject and template_body:
                        st.success(f"Template '{template_name}' created successfully!")
                    else:
                        st.error("Please fill in all required fields")
        
        # Existing templates
        st.subheader("Existing Templates")
        
        templates = [
            {
                "name": "User Welcome",
                "category": "User Management",
                "subject": "Welcome to AutoERP, {user_name}!",
                "body": "Dear {user_name}, welcome to AutoERP. Your account has been created successfully.",
                "variables": ["user_name", "login_url"],
                "usage_count": 45,
                "last_used": "2024-12-10"
            },
            {
                "name": "Security Alert",
                "category": "Security",
                "subject": "Security Alert: {alert_type}",
                "body": "A security event has been detected: {alert_description} at {timestamp}",
                "variables": ["alert_type", "alert_description", "timestamp"],
                "usage_count": 12,
                "last_used": "2024-12-09"
            },
            {
                "name": "System Maintenance",
                "category": "System",
                "subject": "Scheduled Maintenance: {maintenance_date}",
                "body": "System maintenance is scheduled for {maintenance_date} from {start_time} to {end_time}.",
                "variables": ["maintenance_date", "start_time", "end_time"],
                "usage_count": 8,
                "last_used": "2024-12-05"
            }
        ]
        
        for i, template in enumerate(templates):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
                
                with col1:
                    st.write(f"**{template['name']}**")
                    st.caption(f"Category: {template['category']}")
                    st.caption(f"Variables: {', '.join(template['variables'])}")
                
                with col2:
                    st.write(f"Subject: {template['subject']}")
                    st.caption(f"Usage: {template['usage_count']} times")
                
                with col3:
                    st.caption(f"Last used: {template['last_used']}")
                
                with col4:
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("📝 Use", key=f"use_template_{i}"):
                            st.session_state.selected_template = template
                            st.info(f"Template '{template['name']}' selected for use!")
                    
                    with action_col2:
                        if st.button("✏️ Edit", key=f"edit_template_{i}"):
                            st.info(f"Edit template '{template['name']}' - Coming soon!")
                    
                    with action_col3:
                        if st.button("🗑️ Delete", key=f"delete_template_{i}"):
                            st.warning(f"Delete template '{template['name']}' - Coming soon!")
                
                # Template preview
                with st.expander(f"Preview: {template['name']}"):
                    st.markdown("**Subject:**")
                    st.code(template['subject'])
                    st.markdown("**Body:**")
                    st.code(template['body'])
                
                st.divider()
    
    def _render_notification_settings(self):
        """Render notification settings."""
        st.subheader("Notification Settings")
        
        # General settings
        st.markdown("### General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_sender_name = st.text_input("Default Sender Name", value="AutoERP System")
            default_sender_email = st.text_input("Default Sender Email", value="noreply@autoerp.com")
            max_retry_attempts = st.number_input("Max Retry Attempts", value=3, min_value=1, max_value=10)
        
        with col2:
            enable_notifications = st.checkbox("Enable Notifications", value=True)
            log_all_notifications = st.checkbox("Log All Notifications", value=True)
            batch_processing = st.checkbox("Enable Batch Processing", value=True)
        
        # Email settings
        st.markdown("### Email Settings")
        
        with st.expander("Email Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
                use_tls = st.checkbox("Use TLS", value=True)
            
            with col2:
                smtp_username = st.text_input("SMTP Username")
                smtp_password = st.text_input("SMTP Password", type="password")
                use_ssl = st.checkbox("Use SSL", value=False)
            
            if st.button("Test Email Configuration"):
                with st.spinner("Testing email configuration..."):
                    time.sleep(2)
                    st.success("Email configuration test successful!")
        
        # SMS settings
        st.markdown("### SMS Settings")
        
        with st.expander("SMS Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                sms_provider = st.selectbox("SMS Provider", ["Twilio", "AWS SNS", "Custom"])
                sms_api_key = st.text_input("API Key", type="password")
            
            with col2:
                sms_from_number = st.text_input("From Number", placeholder="+1234567890")
                sms_enabled = st.checkbox("Enable SMS Notifications", value=False)
            
            if st.button("Test SMS Configuration"):
                st.info("SMS configuration test - Coming soon!")
        
        # Push notification settings
        st.markdown("### Push Notification Settings")
        
        with st.expander("Push Notification Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                push_service = st.selectbox("Push Service", ["Firebase", "OneSignal", "Custom"])
                push_api_key = st.text_input("Push API Key", type="password")
            
            with col2:
                push_enabled = st.checkbox("Enable Push Notifications", value=False)
                push_sound = st.checkbox("Enable Sound", value=True)
        
        # Notification rules
        st.markdown("### Notification Rules")
        
        with st.expander("Delivery Rules"):
            st.markdown("**Priority-based Delivery Rules:**")
            
            priority_rules = [
                {"priority": "Critical", "email": True, "sms": True, "push": True, "delay": "Immediate"},
                {"priority": "High", "email": True, "sms": False, "push": True, "delay": "1 minute"},
                {"priority": "Medium", "email": True, "sms": False, "push": False, "delay": "5 minutes"},
                {"priority": "Low", "email": True, "sms": False, "push": False, "delay": "15 minutes"}
            ]
            
            for rule in priority_rules:
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                
                with col1:
                    st.write(f"**{rule['priority']}**")
                
                with col2:
                    st.checkbox("Email", value=rule['email'], key=f"{rule['priority']}_email")
                
                with col3:
                    st.checkbox("SMS", value=rule['sms'], key=f"{rule['priority']}_sms")
                
                with col4:
                    st.checkbox("Push", value=rule['push'], key=f"{rule['priority']}_push")
                
                with col5:
                    st.selectbox("Delay", ["Immediate", "1 minute", "5 minutes", "15 minutes"], 
                               index=["Immediate", "1 minute", "5 minutes", "15 minutes"].index(rule['delay']),
                               key=f"{rule['priority']}_delay")
        
        # User preferences
        st.markdown("### User Notification Preferences")
        
        with st.expander("Default User Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                default_email_notifications = st.checkbox("Email Notifications (Default)", value=True)
                default_sms_notifications = st.checkbox("SMS Notifications (Default)", value=False)
                default_push_notifications = st.checkbox("Push Notifications (Default)", value=True)
            
            with col2:
                quiet_hours_enabled = st.checkbox("Enable Quiet Hours", value=True)
                if quiet_hours_enabled:
                    quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
                    quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("08:00", "%H:%M").time())
        
        # Save settings
        if st.button("💾 Save All Settings", type="primary"):
            with st.spinner("Saving notification settings..."):
                time.sleep(2)
                st.success("Notification settings saved successfully!")


# ==================== SETTINGS ====================

class SettingsPage:
    """Settings page implementation."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render settings page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>⚙️ System Settings</h1>
            <p>Configure system parameters and preferences</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏢 General", "🔒 Security", "🗄️ Database", "📧 Email", "🎨 Interface"])
        
        with tab1:
            self._render_general_settings()
        
        with tab2:
            self._render_security_settings()
        
        with tab3:
            self._render_database_settings()
        
        with tab4:
            self._render_email_settings()
        
        with tab5:
            self._render_interface_settings()
    
    def _render_general_settings(self):
        """Render general settings."""
        st.subheader("General System Settings")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### System Information")
            system_info = {
                "System Name": "AutoERP",
                "Version": "1.0.0",
                "Environment": "Production",
                "Uptime": "15 days, 3 hours",
                "Last Restart": "2024-11-25 14:30:00"
            }
            
            for key, value in system_info.items():
                col_a, col_b = st.columns([2, 3])
                with col_a:
                    st.write(f"**{key}:**")
                with col_b:
                    st.write(value)
        
        with col2:
            st.markdown("#### System Configuration")
            
            system_name = st.text_input("System Name", value="AutoERP")
            company_name = st.text_input("Company Name", value="Your Company")
            timezone = st.selectbox("Timezone", [
                "UTC", "America/New_York", "America/Los_Angeles", 
                "Europe/London", "Europe/Paris", "Asia/Tokyo"
            ])
            language = st.selectbox("Default Language", ["English", "French", "Spanish", "German"])
        
        # Business settings
        st.markdown("#### Business Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fiscal_year_start = st.selectbox("Fiscal Year Start", [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
        
        with col2:
            default_currency = st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY", "CAD"])
            decimal_places = st.number_input("Decimal Places", value=2, min_value=0, max_value=6)
        
        with col3:
            auto_backup = st.checkbox("Auto Backup", value=True)
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        
        # Maintenance settings
        st.markdown("#### Maintenance Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            maintenance_mode = st.checkbox("Maintenance Mode", value=False)
            if maintenance_mode:
                maintenance_message = st.text_area(
                    "Maintenance Message",
                    value="System is under maintenance. Please try again later."
                )
        
        with col2:
            debug_mode = st.checkbox("Debug Mode", value=False)
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        
        if st.button("💾 Save General Settings", type="primary"):
            with st.spinner("Saving settings..."):
                time.sleep(2)
                st.success("General settings saved successfully!")
    
    def _render_security_settings(self):
        """Render security settings."""
        st.subheader("Security Settings")
        
        # Password policy
        st.markdown("#### Password Policy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_password_length = st.number_input("Minimum Password Length", value=8, min_value=4, max_value=32)
            require_uppercase = st.checkbox("Require Uppercase", value=True)
            require_lowercase = st.checkbox("Require Lowercase", value=True)
        
        with col2:
            require_numbers = st.checkbox("Require Numbers", value=True)
            require_symbols = st.checkbox("Require Symbols", value=True)
            password_expiry_days = st.number_input("Password Expiry (days)", value=90, min_value=0)
        
        # Session settings
        st.markdown("#### Session Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            session_timeout = st.number_input("Session Timeout (minutes)", value=60, min_value=5, max_value=480)
            max_concurrent_sessions = st.number_input("Max Concurrent Sessions", value=3, min_value=1)
        
        with col2:
            force_logout_inactive = st.checkbox("Force Logout Inactive Users", value=True)
            remember_me_duration = st.number_input("Remember Me Duration (days)", value=30, min_value=1)
        
        # Login security
        st.markdown("#### Login Security")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_login_attempts = st.number_input("Max Login Attempts", value=5, min_value=1, max_value=20)
            lockout_duration = st.number_input("Account Lockout Duration (minutes)", value=30, min_value=1)
        
        with col2:
            enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=False)
            ip_whitelist = st.text_area("IP Whitelist (one per line)", placeholder="192.168.1.0/24\n10.0.0.0/8")
        
        # Security monitoring
        st.markdown("#### Security Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_failed_logins = st.checkbox("Log Failed Logins", value=True)
            alert_security_events = st.checkbox("Alert on Security Events", value=True)
        
        with col2:
            audit_user_actions = st.checkbox("Audit User Actions", value=True)
            encrypt_sensitive_data = st.checkbox("Encrypt Sensitive Data", value=True)
        
        # Security status
        st.markdown("#### Security Status")
        
        security_checks = [
            ("SSL Certificate", "✅ Valid", "success"),
            ("Database Encryption", "✅ Enabled", "success"), 
            ("Backup Encryption", "✅ Enabled", "success"),
            ("Firewall Status", "⚠️ Partially Configured", "warning"),
            ("Intrusion Detection", "❌ Disabled", "error")
        ]
        
        for check, status, status_type in security_checks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{check}:**")
            with col2:
                if status_type == "success":
                    st.success(status)
                elif status_type == "warning":
                    st.warning(status)
                else:
                    st.error(status)
        
        if st.button("💾 Save Security Settings", type="primary"):
            st.success("Security settings saved successfully!")
    
    def _render_database_settings(self):
        """Render database settings."""
        st.subheader("Database Settings")
        
        # Current database info
        st.markdown("#### Current Database Information")
        
        db_info = {
            "Database Engine": "SQLite",
            "Database File": "/path/to/autoerp.db",
            "File Size": "45.2 MB",
            "Tables": "12",
            "Last Backup": "2024-12-10 02:00:00"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(db_info.items())[:3]:
                st.metric(key, value)
        
        with col2:
            for key, value in list(db_info.items())[3:]:
                st.metric(key, value)
        
        # Database configuration
        st.markdown("#### Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            db_engine = st.selectbox("Database Engine", ["SQLite", "PostgreSQL", "MySQL"])
            
            if db_engine != "SQLite":
                db_host = st.text_input("Database Host", value="localhost")
                db_port = st.number_input("Database Port", value=5432 if db_engine == "PostgreSQL" else 3306)
                db_name = st.text_input("Database Name", value="autoerp")
        
        with col2:
            if db_engine != "SQLite":
                db_username = st.text_input("Username")
                db_password = st.text_input("Password", type="password")
                
            connection_pool_size = st.number_input("Connection Pool Size", value=10, min_value=1, max_value=100)
        
        # Backup settings
        st.markdown("#### Backup Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_backup_enabled = st.checkbox("Enable Automatic Backups", value=True)
            backup_frequency = st.selectbox("Backup Frequency", ["Hourly", "Daily", "Weekly"])
            backup_retention = st.number_input("Backup Retention (days)", value=30, min_value=1)
        
        with col2:
            backup_location = st.text_input("Backup Location", value="/backups/")
            compress_backups = st.checkbox("Compress Backups", value=True)
            encrypt_backups = st.checkbox("Encrypt Backups", value=True)
        
        # Maintenance operations
        st.markdown("#### Database Maintenance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Test Connection"):
                with st.spinner("Testing database connection..."):
                    time.sleep(2)
                    st.success("Database connection successful!")
        
        with col2:
            if st.button("💾 Backup Now"):
                with st.spinner("Creating database backup..."):
                    time.sleep(3)
                    st.success("Database backup created successfully!")
        
        with col3:
            if st.button("🧹 Optimize Database"):
                with st.spinner("Optimizing database..."):
                    time.sleep(4)
                    st.success("Database optimization completed!")
        
        with col4:
            if st.button("📊 Database Stats"):
                st.info("Database statistics - Coming soon!")
        
        # Database health
        st.markdown("#### Database Health")
        
        health_metrics = [
            ("Connection Status", "✅ Connected", "success"),
            ("Performance", "🟢 Good", "success"),
            ("Storage Usage", "⚠️ 75% Full", "warning"),
            ("Last Maintenance", "✅ Yesterday", "success")
        ]
        
        cols = st.columns(4)
        for i, (metric, status, status_type) in enumerate(health_metrics):
            with cols[i]:
                st.metric(metric, status)
        
        if st.button("💾 Save Database Settings", type="primary"):
            st.success("Database settings saved successfully!")
    
    def _render_email_settings(self):
        """Render email settings."""
        st.subheader("Email Configuration")
        
        # SMTP settings
        st.markdown("#### SMTP Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smtp_enabled = st.checkbox("Enable Email", value=True)
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
        
        with col2:
            smtp_username = st.text_input("SMTP Username")
            smtp_password = st.text_input("SMTP Password", type="password")
            use_authentication = st.checkbox("Use Authentication", value=True)
        
        # Security settings
        col1, col2 = st.columns(2)
        
        with col1:
            use_tls = st.checkbox("Use TLS", value=True)
            use_ssl = st.checkbox("Use SSL", value=False)
        
        with col2:
            timeout = st.number_input("Connection Timeout (seconds)", value=30, min_value=5)
            max_retries = st.number_input("Max Retry Attempts", value=3, min_value=1)
        
        # Default email settings
        st.markdown("#### Default Email Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            from_name = st.text_input("From Name", value="AutoERP System")
            from_email = st.text_input("From Email", value="noreply@autoerp.com")
        
        with col2:
            reply_to = st.text_input("Reply To Email")
            bounce_email = st.text_input("Bounce Handling Email")
        
        # Email templates
        st.markdown("#### Email Template Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_footer = st.text_area(
                "Default Email Footer",
                value="This email was sent by AutoERP System. Please do not reply to this email.",
                height=100
            )
        
        with col2:
            company_logo_url = st.text_input("Company Logo URL")
            company_address = st.text_area(
                "Company Address",
                value="123 Business St.\nCity, State 12345\nCountry",
                height=100
            )
        
        # Email testing
        st.markdown("#### Test Email Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            test_email = st.text_input("Test Email Address", placeholder="test@example.com")
        
        with col2:
            if st.button("📧 Send Test Email"):
                if test_email:
                    with st.spinner("Sending test email..."):
                        time.sleep(3)
                        st.success(f"Test email sent to {test_email}!")
                else:
                    st.error("Please enter a test email address")
        
        # Email statistics
        st.markdown("#### Email Statistics (Last 30 Days)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Emails Sent", "1,234")
        
        with col2:
            st.metric("Delivery Rate", "98.5%")
        
        with col3:
            st.metric("Bounce Rate", "1.2%")
        
        with col4:
            st.metric("Failed Deliveries", "18")
        
        if st.button("💾 Save Email Settings", type="primary"):
            st.success("Email settings saved successfully!")
    
    def _render_interface_settings(self):
        """Render interface settings."""
        st.subheader("User Interface Settings")
        
        # Theme settings
        st.markdown("#### Theme & Appearance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Color Theme", ["Light", "Dark", "Auto"])
            primary_color = st.color_picker("Primary Color", value="#1f77b4")
            secondary_color = st.color_picker("Secondary Color", value="#ff7f0e")
        
        with col2:
            sidebar_position = st.selectbox("Sidebar Position", ["Left", "Right"])
            enable_animations = st.checkbox("Enable Animations", value=True)
            compact_mode = st.checkbox("Compact Mode", value=False)
        
        # Layout settings
        st.markdown("#### Layout Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_page_size = st.selectbox("Default Page Size", [10, 25, 50, 100])
            show_tooltips = st.checkbox("Show Tooltips", value=True)
            auto_refresh = st.checkbox("Auto Refresh Data", value=True)
        
        with col2:
            if auto_refresh:
                refresh_interval = st.number_input("Refresh Interval (seconds)", value=30, min_value=5)
            
            enable_keyboard_shortcuts = st.checkbox("Enable Keyboard Shortcuts", value=True)
            show_breadcrumbs = st.checkbox("Show Breadcrumbs", value=True)
        
        # Dashboard settings
        st.markdown("#### Dashboard Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dashboard_layout = st.selectbox("Dashboard Layout", ["Grid", "List", "Cards"])
            show_welcome_message = st.checkbox("Show Welcome Message", value=True)
        
        with col2:
            default_chart_type = st.selectbox("Default Chart Type", ["Line", "Bar", "Pie", "Area"])
            chart_animation = st.checkbox("Chart Animations", value=True)
        
        # Accessibility settings
        st.markdown("#### Accessibility Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_contrast = st.checkbox("High Contrast Mode", value=False)
            large_fonts = st.checkbox("Large Fonts", value=False)
            screen_reader_support = st.checkbox("Screen Reader Support", value=True)
        
        with col2:
            keyboard_navigation = st.checkbox("Enhanced Keyboard Navigation", value=True)
            focus_indicators = st.checkbox("Enhanced Focus Indicators", value=True)
            alt_text_images = st.checkbox("Alt Text for Images", value=True)
        
        # Language and localization
        st.markdown("#### Language & Localization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            interface_language = st.selectbox("Interface Language", ["English", "French", "Spanish", "German"])
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
        
        with col2:
            time_format = st.selectbox("Time Format", ["12 Hour", "24 Hour"])
            number_format = st.selectbox("Number Format", ["1,234.56", "1.234,56", "1 234.56"])
        
        # Preview section
        st.markdown("#### Preview")
        
        # Show a preview of current settings
        preview_data = {
            "Date": ["2024-12-10", "2024-12-09", "2024-12-08"],
            "Value": [1234.56, 2345.67, 3456.78],
            "Status": ["Active", "Pending", "Completed"]
        }
        
        preview_df = pd.DataFrame(preview_data)
        
        st.markdown("**Sample Data Table:**")
        st.dataframe(preview_df, use_container_width=True)
        
        # Sample chart
        fig = px.line(preview_df, x="Date", y="Value", title="Sample Chart with Current Settings")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("💾 Save Interface Settings", type="primary"):
            st.success("Interface settings saved successfully!")


# ==================== SYSTEM HEALTH ====================

class SystemHealthPage:
    """System health monitoring page."""
    
    def __init__(self, app: AutoERPApplication):
        self.app = app
    
    def render(self):
        """Render system health page."""
        st.markdown("""
        <div class="dashboard-header">
            <h1>❤️ System Health</h1>
            <p>Monitor system performance and health metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto refresh
        col1, col2 = st.columns([1, 4])
        
        with col1:
            auto_refresh = st.checkbox("Auto Refresh (10s)")
        
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        
        # Get system health data
        with st.spinner("Loading health metrics..."):
            health_data = asyncio.run(self._get_health_data())
        
        # Overall status
        overall_status = health_data['overall_status']
        status_color = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴',
            'unknown': '⚪'
        }.get(overall_status, '⚪')
        
        st.markdown(f"## {status_color} System Status: {overall_status.title()}")
        
        # Health metrics cards
        self._render_health_metrics(health_data)
        
        # Detailed monitoring
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🗄️ Database", "💾 Storage", "🌐 Network"])
        
        with tab1:
            self._render_performance_monitoring(health_data)
        
        with tab2:
            self._render_database_monitoring(health_data)
        
        with tab3:
            self._render_storage_monitoring(health_data)
        
        with tab4:
            self._render_network_monitoring(health_data)
    
    async def _get_health_data(self) -> Dict[str, Any]:
        """Get comprehensive system health data."""
        try:
            health_status = await self.app.get_health_status()
            app_metrics = self.app.get_metrics()
            
            # Simulate additional health metrics
            cpu_usage = np.random.uniform(20, 80)
            memory_usage = np.random.uniform(30, 90) 
            disk_usage = np.random.uniform(40, 85)
            
            # Determine overall status
            if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
                overall_status = 'critical'
            elif cpu_usage > 75 or memory_usage > 85 or disk_usage > 85:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            return {
                'overall_status': overall_status,
                'health_status': health_status,
                'app_metrics': app_metrics,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'uptime_seconds': app_metrics.get('uptime_seconds', 0),
                'active_connections': np.random.randint(10, 50),
                'request_rate': np.random.uniform(10, 100),
                'error_rate': np.random.uniform(0, 5),
                'response_time': np.random.uniform(50, 500)
            }
        
        except Exception as e:
            logger.error(f"Error getting health data: {e}")
            return {
                'overall_status': 'unknown',
                'error': str(e)
            }
    
    def _render_health_metrics(self, health_data: Dict[str, Any]):
        """Render health metrics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        # System uptime
        with col1:
            uptime_hours = health_data.get('uptime_seconds', 0) / 3600
            st.metric(
                "System Uptime",
                f"{uptime_hours:.1f} hours",
                delta="Running" if uptime_hours > 0 else "Offline"
            )
        
        # CPU usage
        with col2:
            cpu_usage = health_data.get('cpu_usage', 0)
            cpu_delta = "Normal" if cpu_usage < 75 else "High"
            st.metric(
                "CPU Usage",
                f"{cpu_usage:.1f}%",
                delta=cpu_delta
            )
        
        # Memory usage  
        with col3:
            memory_usage = health_data.get('memory_usage', 0)
            memory_delta = "Normal" if memory_usage < 85 else "High"
            st.metric(
                "Memory Usage",
                f"{memory_usage:.1f}%", 
                delta=memory_delta
            )
        
        # Active connections
        with col4:
            connections = health_data.get('active_connections', 0)
            st.metric(
                "Active Connections",
                str(connections),
                delta=f"+{np.random.randint(1, 5)}"
            )
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            request_rate = health_data.get('request_rate', 0)
            st.metric("Request Rate", f"{request_rate:.1f}/sec")
        
        with col2:
            error_rate = health_data.get('error_rate', 0)
            error_status = "Good" if error_rate < 1 else "High"
            st.metric("Error Rate", f"{error_rate:.2f}%", delta=error_status)
        
        with col3:
            response_time = health_data.get('response_time', 0)
            response_status = "Fast" if response_time < 200 else "Slow"
            st.metric("Avg Response Time", f"{response_time:.0f}ms", delta=response_status)
        
        with col4:
            disk_usage = health_data.get('disk_usage', 0)
            disk_status = "Normal" if disk_usage < 85 else "High"
            st.metric("Disk Usage", f"{disk_usage:.1f}%", delta=disk_status)
    
    def _render_performance_monitoring(self, health_data: Dict[str, Any]):
        """Render performance monitoring section."""
        st.subheader("Performance Monitoring")
        
        # Generate time series data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), periods=60, freq='1min')
        
        perf_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu': np.random.uniform(20, 80, 60),
            'memory': np.random.uniform(30, 90, 60),
            'requests': np.random.poisson(50, 60),
            'response_time': np.random.exponential(100, 60)
        })
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU and Memory usage
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=perf_data['timestamp'],
                y=perf_data['cpu'],
                mode='lines',
                name='CPU %',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=perf_data['timestamp'],
                y=perf_data['memory'],
                mode='lines',
                name='Memory %',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='CPU & Memory Usage (Last Hour)',
                yaxis_title='Usage (%)',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Request rate and response time
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['requests'], name='Requests/min'),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['response_time'], name='Response Time (ms)'),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Requests/min", secondary_y=False)
            fig.update_yaxes(title_text="Response Time (ms)", secondary_y=True)
            
            fig.update_layout(title_text="Request Rate & Response Time", height=300)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance alerts
        st.subheader("Performance Alerts")
        
        alerts = [
            {"severity": "Warning", "message": "CPU usage above 75% for 5 minutes", "time": "2 min ago"},
            {"severity": "Info", "message": "Memory usage returned to normal levels", "time": "15 min ago"},
            {"severity": "Critical", "message": "Response time spike detected", "time": "1 hour ago"}
        ]
        
        for alert in alerts:
            severity_color = {"Critical": "🔴", "Warning": "🟡", "Info": "🔵"}.get(alert["severity"], "⚪")
            
            st.markdown(f"""
            <div style="
                border-left: 4px solid {'#dc3545' if alert['severity'] == 'Critical' else '#ffc107' if alert['severity'] == 'Warning' else '#17a2b8'};
                padding: 10px;
                margin: 5px 0;
                background-color: #f8f9fa;
            ">
                {severity_color} <strong>{alert['severity']}:</strong> {alert['message']} <em>({alert['time']})</em>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_database_monitoring(self, health_data: Dict[str, Any]):
        """Render database monitoring section."""
        st.subheader("Database Health")
        
        # Database status
        db_status = health_data.get('health_status', {}).get('components', {}).get('database', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = db_status.get('status', 'unknown')
            status_icon = "✅" if status == 'healthy' else "❌"
            st.metric("Database Status", f"{status_icon} {status.title()}")
        
        with col2:
            response_time = db_status.get('response_time_ms', 0)
            st.metric("Response Time", f"{response_time:.1f}ms")
        
        with col3:
            st.metric("Active Connections", "8/20")
        
        # Database performance metrics
        db_metrics = pd.DataFrame({
            'Metric': ['Queries/sec', 'Slow Queries', 'Connections', 'Cache Hit Rate', 'Index Usage'],
            'Current': [45.2, 0, 8, 94.5, 87.2],
            'Threshold': [100, 5, 20, 90, 80],
            'Status': ['Good', 'Good', 'Good', 'Good', 'Good']
        })
        
        st.dataframe(db_metrics, use_container_width=True, hide_index=True)
        
        # Query performance
        st.subheader("Query Performance")
        
        query_data = pd.DataFrame({
            'Query Type': ['SELECT', 'INSERT', 'UPDATE', 'DELETE'],
            'Count': [1250, 45, 23, 12],
            'Avg Duration (ms)': [12.5, 45.2, 67.8, 23.1],
            'Max Duration (ms)': [456, 1234, 2345, 567]
        })
        
        fig = px.bar(query_data, x='Query Type', y='Count', title='Query Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_storage_monitoring(self, health_data: Dict[str, Any]):
        """Render storage monitoring section."""
        st.subheader("Storage Health")
        
        # Storage metrics
        storage_data = [
            {"Mount": "/", "Total": "100 GB", "Used": "65 GB", "Available": "35 GB", "Usage": 65},
            {"Mount": "/var", "Total": "50 GB", "Used": "20 GB", "Available": "30 GB", "Usage": 40},
            {"Mount": "/backup", "Total": "500 GB", "Used": "200 GB", "Available": "300 GB", "Usage": 40}
        ]
        
        for storage in storage_data:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"**{storage['Mount']}**")
            
            with col2:
                st.write(storage['Total'])
            
            with col3:
                st.write(storage['Used'])
            
            with col4:
                st.write(storage['Available'])
            
            with col5:
                usage_color = "red" if storage['Usage'] > 85 else "orange" if storage['Usage'] > 70 else "green"
                st.markdown(f"<div style='color: {usage_color}'><strong>{storage['Usage']}%</strong></div>", 
                          unsafe_allow_html=True)
        
        # Storage usage chart
        fig = go.Figure(data=[
            go.Bar(name='Used', x=[s['Mount'] for s in storage_data], y=[s['Usage'] for s in storage_data]),
            go.Bar(name='Available', x=[s['Mount'] for s in storage_data], 
                  y=[100 - s['Usage'] for s in storage_data])
        ])
        
        fig.update_layout(
            title='Storage Usage by Mount Point',
            barmode='stack',
            yaxis_title='Usage (%)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_network_monitoring(self, health_data: Dict[str, Any]):
        """Render network monitoring section."""
        st.subheader("Network Health")
        
        # Network metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Network In", "1.2 MB/s")
        
        with col2:
            st.metric("Network Out", "850 KB/s")
        
        with col3:
            st.metric("Packet Loss", "0.01%")
        
        with col4:
            st.metric("Latency", "15ms")
        
        # Network traffic chart
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), periods=60, freq='1min')
        
        network_data = pd.DataFrame({
            'timestamp': timestamps,
            'incoming': np.random.uniform(0.5, 2.0, 60),
            'outgoing': np.random.uniform(0.3, 1.5, 60)
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=network_data['timestamp'],
            y=network_data['incoming'],
            mode='lines',
            name='Incoming (MB/s)',
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=network_data['timestamp'],
            y=network_data['outgoing'],
            mode='lines',
            name='Outgoing (MB/s)',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Network Traffic (Last Hour)',
            yaxis_title='Traffic (MB/s)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN APPLICATION ====================

class AutoERPStreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.config = AutoERPConfig()
        self.app_instance = None
        self.navigation_manager = NavigationManager()
        self.auth_manager = None
        self.pages = {}
    
    async def initialize(self):
        """Initialize the application."""
        try:
            self.app_instance = AutoERPApplication(self.config)
            await self.app_instance.initialize()
            
            self.auth_manager = AuthenticationManager(self.app_instance)
            
            # Initialize pages
            self.pages = {
                'dashboard': DashboardPage(self.app_instance),
                'users': UserManagementPage(self.app_instance),
                'data': DataManagementPage(self.app_instance),
                'reports': ReportsPage(self.app_instance),
                'notifications': NotificationsPage(self.app_instance),
                'settings': SettingsPage(self.app_instance),
                'health': SystemHealthPage(self.app_instance)
            }
            
            st.session_state.app_instance = self.app_instance
            
        except Exception as e:
            st.error(f"Failed to initialize application: {e}")
            logger.error(f"Application initialization error: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        # Configure Streamlit
        StreamlitConfig.configure_page()
        
        # Initialize session state
        SessionState.initialize()
        
        # Initialize app if not already done
        if st.session_state.app_instance is None:
            with st.spinner("Initializing AutoERP system..."):
                asyncio.run(self.initialize())
        
        # Check session expiration
        if SessionState.is_session_expired() and st.session_state.authenticated:
            st.warning("Your session has expired. Please login again.")
            SessionState.clear()
            SessionState.initialize()
        
        # Render navigation
        self.navigation_manager.render_sidebar()
        
        # Main content area
        if not st.session_state.authenticated:
            # Show login page
            self.auth_manager.render_login_form()
        else:
            # Show main application
            current_page = self.navigation_manager.get_current_page()
            
            if current_page in self.pages:
                try:
                    self.pages[current_page].render()
                except Exception as e:
                    st.error(f"Error loading page: {e}")
                    logger.error(f"Page rendering error for {current_page}: {e}")
            else:
                st.error(f"Page '{current_page}' not found")


# ==================== FLASK SOCKETIO INTEGRATION ====================

class AutoERPFlaskApp:
    """Flask application for real-time features."""
    
    def __init__(self, autoerp_app: AutoERPApplication):
        self.autoerp_app = autoerp_app
        self.flask_app = Flask(__name__)
        self.flask_app.secret_key = secrets.token_urlsafe(32)
        
        self.socketio = SocketIO(
            self.flask_app,
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )
        
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.flask_app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            try:
                health = asyncio.run(self.autoerp_app.get_health_status())
                return jsonify(health)
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.flask_app.route('/api/metrics')
        def get_metrics():
            """Get application metrics."""
            try:
                metrics = self.autoerp_app.get_metrics()
                return jsonify(metrics)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _setup_socket_events(self):
        """Setup Socket.IO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'status': 'Connected to AutoERP real-time service'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            """Handle room joining."""
            room = data.get('room')
            if room:
                join_room(room)
                emit('joined_room', {'room': room})
        
        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            """Handle room leaving."""
            room = data.get('room')
            if room:
                leave_room(room)
                emit('left_room', {'room': room})
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the Flask-SocketIO application."""
        self.socketio.run(
            self.flask_app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )

def dashboard_page():
    st.title("🏢 AutoERP Dashboard")
    st.write("Welcome to AutoERP!")


def tables_page():
    st.title("Tables")
    st.write("Data tables view")
    
# ==================== ENTRY POINT ====================

def main():
    """Main entry point for the AutoERP UI application."""
    try:
        # Initialize and run Streamlit app
        app = AutoERPStreamlitApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application startup error: {e}")
        logger.error(f"Application startup error: {e}")


if __name__ == "__main__":
    main()