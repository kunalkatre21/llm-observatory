import streamlit as st
import plotly.graph_objects as go
import pandas as pd

class TestVisualizer:
    def create_test_report(self, results):
        st.subheader("📊 Detailed Test Report")

        # Test Suite Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Tests Passed",
                f"{results['passed_individual_tests']}/{results['total_individual_tests']}"
            )
        with col2:
            st.metric(
                "Avg Response Time",
                f"{results['avg_time']:.2f}s"
            )
        with col3:
            st.metric(
                "Reliability Score",
                f"{results['reliability_score']:.1f}%"
            )

        # Detailed Category Breakdown
        st.write("### Test Categories Performance")

        for category, result in results['category_results'].items():
            with st.expander(f"🔍 {category.replace('_', ' ').title()}", expanded=True):
                if result['passed']:
                    st.success("All tests passed!")
                else:
                    failed_count = len([x for x in result['details'] if not x])
                    total_count = len(result['details'])
                    st.error(f"Failed {failed_count}/{total_count} tests")

        # Performance trends
        st.write("### 📈 Performance Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            # Convert range to list 
            x=list(range(results['total_individual_tests'])),
            y=[m["time"] for m in results['performance_metrics']['response_times']],
            name="Response Time"            
        ))
        st.plotly_chart(fig)
