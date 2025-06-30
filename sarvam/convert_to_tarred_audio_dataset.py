    def _read_manifest_parallel(self, manifest_path: str, config: ASRTarredDatasetConfig, num_workers: int = 100):
        """Read and filters data from the manifest"""
        # Read the existing manifest
        import multiprocessing as mp
        from functools import partial
                
        # Read all lines from manifest
        with open(manifest_path, 'r', encoding='utf-8') as m:
            lines = m.readlines()
            
        print(f"Processing {len(lines)} entries with {num_workers} workers")
                
        # Create a pool of workers
        pool = mp.Pool(processes=num_workers)
        
        # Process entries in parallel
        process_func = partial(self._process_entry, manifest_path=manifest_path, config=config)
        results = []
        
        # Process in chunks to show progress
        chunk_size = max(1, len(lines) // 100)
        print(f"Using chunk_size: {chunk_size}")
        for i, result in enumerate(pool.imap(process_func, lines, chunksize=chunk_size)):
            results.append(result)
            if (i + 1) % 100000 == 0:
                print(f"counter: {i + 1}")
        
        pool.close()
        pool.join()
        
        # Combine results
        entries = []
        filtered_entries = []
        total_duration = 0.0
        filtered_duration = 0.0
        
        for entry, entry_duration, filtered_entry, filtered_entry_duration in results:
            if entry is not None:
                entries.append(entry)
                total_duration += entry_duration
            if filtered_entry is not None:
                filtered_entries.append(filtered_entry)
                filtered_duration += filtered_entry_duration
                
        return entries, total_duration, filtered_entries, filtered_duration
