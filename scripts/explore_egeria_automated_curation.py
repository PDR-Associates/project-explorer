#!/usr/bin/env python3
"""
Exploration script to understand Egeria's AutomatedCuration API for PostgreSQL.

This script demonstrates how to:
1. Find PostgreSQL technology types
2. Find catalog templates for PostgreSQL
3. Find governance action processes for PostgreSQL
4. Understand the workflow for surveying databases

Goal: Understand Egeria's capabilities before building custom surveyors.
"""

import json
from pyegeria import AutomatedCuration
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table

console = Console()


def explore_postgresql_capabilities():
    """Comprehensive exploration of PostgreSQL support in Egeria."""
    
    console.print("\n[bold cyan]═══ Exploring PostgreSQL Capabilities in Egeria ═══[/bold cyan]\n")
    
    # Connect to Egeria
    # Note: Use host.docker.internal:9443 when running in Docker/Jupyter
    # Use hedwig.local:9443 or localhost:9443 for local connections
    server = "https://host.docker.internal:9443"
    view_server = "qs-view-server"
    user = "erinoverview"
    password = "secret"
    
    console.print(f"Connecting to: {server}")
    console.print(f"View Server: {view_server}")
    console.print(f"User: {user}\n")
    
    try:
        client = AutomatedCuration(view_server, server, user_id=user, user_pwd=password)
        
        # Create bearer token for authentication
        console.print("[yellow]Creating bearer token...[/yellow]")
        client.create_egeria_bearer_token(user, password)
        console.print("[green]✓ Authenticated[/green]\n")
        
        # ═══ Step 1: Technology Types ═══
        console.print("\n[bold cyan]═══ Step 1: Technology Types ═══[/bold cyan]\n")
        search_string = "PostgreSQL"
        console.print(f"Searching for technology types matching '{search_string}'...\n")
        
        tech_types = client.find_technology_types(search_string, starts_with=False, ends_with=False)
        
        if tech_types:
            console.print(f"[green]✓ Found {len(tech_types)} technology type(s)[/green]\n")
            
            # Create summary table
            table = Table(title="PostgreSQL Technology Types")
            table.add_column("Display Name", style="cyan")
            table.add_column("Category", style="yellow")
            table.add_column("GUID", style="green")
            table.add_column("Description", style="white")
            
            for tech_type in tech_types:
                table.add_row(
                    tech_type.get('displayName', 'N/A'),
                    tech_type.get('category', 'N/A'),
                    tech_type.get('technologyTypeGUID', 'N/A')[:8] + "...",
                    tech_type.get('description', 'N/A')[:60] + "..."
                )
            
            console.print(table)
            console.print()
        else:
            console.print("[red]No technology types found![/red]\n")
        
        # ═══ Step 2: Catalog Templates ═══
        console.print("\n[bold cyan]═══ Step 2: Catalog Templates ═══[/bold cyan]\n")
        console.print("Searching for catalog templates for PostgreSQL...\n")
        
        try:
            # Try to find templates - the method name might vary
            templates = client.find_templates(search_string, starts_with=False, ends_with=False)
            
            if templates:
                console.print(f"[green]✓ Found {len(templates)} template(s)[/green]\n")
                for i, template in enumerate(templates, 1):
                    console.print(f"\n[bold]Template {i}:[/bold]")
                    console.print(Panel(JSON(json.dumps(template, indent=2)), title="Template Structure"))
            else:
                console.print("[yellow]No templates found[/yellow]\n")
        except AttributeError as e:
            console.print(f"[yellow]Method 'find_templates' not available: {e}[/yellow]\n")
            console.print("[yellow]Trying alternative approach...[/yellow]\n")
            
            # Try to get templates for a specific technology type
            if tech_types:
                tech_type_guid = tech_types[0].get('technologyTypeGUID')
                console.print(f"Attempting to get templates for GUID: {tech_type_guid}\n")
                try:
                    templates = client.get_templates_for_technology_type(tech_type_guid)
                    if templates:
                        console.print(f"[green]✓ Found {len(templates)} template(s)[/green]\n")
                        console.print(Panel(JSON(json.dumps(templates, indent=2))))
                    else:
                        console.print("[yellow]No templates found for this technology type[/yellow]\n")
                except Exception as e2:
                    console.print(f"[red]Error getting templates: {e2}[/red]\n")
        
        # ═══ Step 3: Governance Action Processes ═══
        console.print("\n[bold cyan]═══ Step 3: Governance Action Processes ═══[/bold cyan]\n")
        console.print("Searching for governance processes related to PostgreSQL...\n")
        
        try:
            # Try to find governance processes
            processes = client.find_governance_action_processes(search_string, starts_with=False, ends_with=False)
            
            if processes:
                console.print(f"[green]✓ Found {len(processes)} process(es)[/green]\n")
                for i, process in enumerate(processes, 1):
                    console.print(f"\n[bold]Process {i}:[/bold]")
                    console.print(Panel(JSON(json.dumps(process, indent=2)), title="Process Structure"))
            else:
                console.print("[yellow]No governance processes found[/yellow]\n")
        except AttributeError as e:
            console.print(f"[yellow]Method 'find_governance_action_processes' not available: {e}[/yellow]\n")
        
        # ═══ Step 4: List Available Methods ═══
        console.print("\n[bold cyan]═══ Step 4: Available AutomatedCuration Methods ═══[/bold cyan]\n")
        console.print("Methods related to templates and processes:\n")
        
        relevant_methods = [m for m in dir(client) if not m.startswith('_') and 
                          ('template' in m.lower() or 'process' in m.lower() or 'governance' in m.lower())]
        
        for method in sorted(relevant_methods):
            console.print(f"  • {method}")
        
        console.print()
        
        # ═══ Summary ═══
        console.print("\n[bold cyan]═══ Summary ═══[/bold cyan]\n")
        console.print("[bold]Key Findings:[/bold]")
        console.print(f"  • Found {len(tech_types) if tech_types else 0} PostgreSQL technology types")
        console.print("  • Technology types are 'Valid Metadata Values' - they define what PostgreSQL assets look like")
        console.print("  • Catalog templates and governance processes are separate entities")
        console.print("  • Need to explore the correct API methods to find templates and processes")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. Identify the correct methods to query templates and processes")
        console.print("  2. Understand the relationship between technology types, templates, and processes")
        console.print("  3. Learn how to trigger a survey for a PostgreSQL database")
        console.print()
        
        client.close_session()
        console.print("[green]✓ Session closed[/green]\n")
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    explore_postgresql_capabilities()

# Made with Bob
