#!/usr/bin/env python3
"""
Exploration script to understand Egeria's AutomatedCuration API.

This script demonstrates how to:
1. Connect to Egeria
2. Find technology types (like PostgreSQL)
3. Examine the structure of technology types (including nested governance processes)
4. Check for existing assets
5. Check for existing surveys

Goal: Understand Egeria's capabilities before building custom surveyors.
"""

import json, os, sys
from pyegeria import AutomatedCuration, EgeriaTech
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()


def explore_technology_types():
    """Step 1: Find and examine PostgreSQL technology type structure."""
    
    console.print("\n[bold cyan]═══ Step 1: Exploring Technology Types ═══[/bold cyan]\n")
    
    # Connect to Egeria
    server = "https://hedwig.local:9443"
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
        
    
        # Search for PostgreSQL technology type
        search_string = "PostgreSQL"
        console.print(f"[yellow]Searching for technology types matching '{search_string}'...[/yellow]\n")
        
        tech_types = client.find_technology_types(search_string, starts_with=False, ends_with=False)
        
        if not tech_types:
            console.print("[red]No technology types found![/red]")
            return
        
        console.print(f"[green]✓ Found {len(tech_types)} technology type(s)[/green]\n")
        
        # Display the full structure
        for i, tech_type in enumerate(tech_types, 1):
            console.print(f"\n[bold]Technology Type {i}:[/bold]")
            console.print(Panel(JSON(json.dumps(tech_type, indent=2)), title="Full Structure"))
            
            # Extract key information
            if isinstance(tech_type, dict):
                console.print("\n[bold cyan]Key Information:[/bold cyan]")
                console.print(f"  • Name: {tech_type.get('name', 'N/A')}")
                console.print(f"  • Category: {tech_type.get('category', 'N/A')}")
                console.print(f"  • GUID: {tech_type.get('guid', 'N/A')}")
                
                # Check for nested governance processes
                if 'governanceActionProcesses' in tech_type:
                    processes = tech_type['governanceActionProcesses']
                    console.print(f"\n[bold green]  ✓ Found {len(processes)} governance process(es):[/bold green]")
                    for proc in processes:
                        console.print(f"    - {proc.get('displayName', proc.get('name', 'Unknown'))}")
                elif 'catalogTemplates' in tech_type:
                    templates = tech_type['catalogTemplates']
                    console.print(f"\n[bold yellow]  • Found {len(templates)} catalog template(s)[/bold yellow]")
                else:
                    console.print("\n[yellow]  ⚠ No governance processes or templates found in this structure[/yellow]")
        
        client.close_session()
        console.print("\n[green]✓ Session closed[/green]")
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    explore_technology_types()
    

# Made with Bob
