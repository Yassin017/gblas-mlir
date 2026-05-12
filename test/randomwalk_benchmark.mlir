#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

module {
    func.func private @start_timer()
    func.func private @stop_timer()

    func.func @randomwalk(%nodes_i32: i32, %edges_i32: i32, %coords_mem: memref<?x2xi64>, %vals_mem: memref<?xf32>, %out_dist: memref<?xf32>) attributes { llvm.emit_c_interface } {
        
        %num_nodes = arith.index_cast %nodes_i32 : i32 to index
        %num_edges = arith.index_cast %edges_i32 : i32 to index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c100 = arith.constant 100 : index 
        
        %f0 = arith.constant 0.0 : f32
        %f1 = arith.constant 1.0 : f32

        // 1. Load Coordinates from MemRef
        %empty_coords = tensor.empty(%num_edges) : tensor<?x2xi64>
        %coords_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_coords = %empty_coords) -> tensor<?x2xi64> {
            %row = memref.load %coords_mem[%e, %c0] : memref<?x2xi64>
            %col = memref.load %coords_mem[%e, %c1] : memref<?x2xi64>
            %t1 = tensor.insert %row into %t_coords[%e, %c0] : tensor<?x2xi64>
            %t2 = tensor.insert %col into %t1[%e, %c1] : tensor<?x2xi64>
            scf.yield %t2 : tensor<?x2xi64>
        }

        // 2. Load Values from MemRef
        %empty_vals = tensor.empty(%num_edges) : tensor<?xf32>
        %vals_tensor = scf.for %e = %c0 to %num_edges step %c1 iter_args(%t_vals = %empty_vals) -> tensor<?xf32> {
            %v = memref.load %vals_mem[%e] : memref<?xf32>
            %t1 = tensor.insert %v into %t_vals[%e] : tensor<?xf32>
            scf.yield %t1 : tensor<?xf32>
        }

        // 3. Generate #CSR sparse transition matrix
        %A = gblas.from_coo %coords_tensor, %vals_tensor (%num_nodes, %num_nodes)
            : tensor<?x2xi64>, tensor<?xf32> -> tensor<?x?xf32, #CSR>

        // 4. Initial State Allocation (1.0 at Node 0, 0.0 elsewhere)
        %empty_f32 = tensor.empty(%num_nodes) : tensor<?xf32>
        %v_start_0 = linalg.fill ins(%f0 : f32) outs(%empty_f32 : tensor<?xf32>) -> tensor<?xf32>
        %v_start = tensor.insert %f1 into %v_start_0[%c0] : tensor<?xf32>

        func.call @start_timer() : () -> () // time start

        // 5. The Random Walk Loop (10 Iterations)
        %v_final = scf.for %i = %c0 to %c100 step %c1 
            iter_args(%v_curr = %v_start) -> (tensor<?xf32>) {
            
            %scratch = tensor.empty(%num_nodes) : tensor<?xf32>
            %zeroed_scratch = linalg.fill ins(%f0 : f32) outs(%scratch : tensor<?xf32>) -> tensor<?xf32>

            // v_computed = v_curr * A
            %v_computed = gblas.vxm %v_curr, %A outs(%zeroed_scratch)
                combine = multiplies 
                reduce = plus 
                : tensor<?xf32>, tensor<?x?xf32, #CSR>, tensor<?xf32> 
                -> tensor<?xf32>

            // --- THE FIX: The Bulletproof Copy Loop ---
            // Safely copy values back into the loop-carried variable
            %v_next = scf.for %j = %c0 to %num_nodes step %c1 
                iter_args(%v_acc = %v_curr) -> (tensor<?xf32>) {
                
                %val = tensor.extract %v_computed[%j] : tensor<?xf32>
                %v_inserted = tensor.insert %val into %v_acc[%j] : tensor<?xf32>
                
                scf.yield %v_inserted : tensor<?xf32>
            }

            scf.yield %v_next : tensor<?xf32>
        }

        func.call @stop_timer() : () -> () // time end

        // 6. Write back to C++ Output MemRef
        scf.for %k = %c0 to %num_nodes step %c1 {
            %v_val = tensor.extract %v_final[%k] : tensor<?xf32>
            memref.store %v_val, %out_dist[%k] : memref<?xf32>
        }

        return
    }
}